#!/usr/bin/env bash
# ResearchOps Agent — GCP 프로젝트 초기 설정 스크립트
# 프로젝트는 이미 생성됨 (researchops-agent-202605)
# GCP Console에서 결제 활성화 후 이 스크립트 실행
set -euo pipefail

# ── 설정 변수 ─────────────────────────────────────────────────────
PROJECT_ID="researchops-agent-202605"
REGION="asia-northeast3"
GITHUB_REPO="FTinMacBook/ResearchOps-Agent"
SA_EMAIL="researchops-sa@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOY_SA="github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com"
# ─────────────────────────────────────────────────────────────────

gcloud config set project "${PROJECT_ID}"

echo "=== [1/5] 필요한 API 활성화 ==="
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    iam.googleapis.com \
    iamcredentials.googleapis.com

echo "=== [2/5] Artifact Registry 저장소 생성 ==="
gcloud artifacts repositories create researchops \
    --repository-format=docker \
    --location="${REGION}" \
    --description="ResearchOps Agent container images" 2>/dev/null \
    || echo "이미 존재함 — 건너뜀"

echo "=== [3/5] Secret Manager에 GOOGLE_API_KEY 등록 ==="
read -rsp "GOOGLE_API_KEY 입력: " GOOGLE_API_KEY
echo ""
echo -n "${GOOGLE_API_KEY}" | gcloud secrets create google-api-key \
    --data-file=- \
    --replication-policy=automatic 2>/dev/null \
    || echo -n "${GOOGLE_API_KEY}" | gcloud secrets versions add google-api-key --data-file=-
unset GOOGLE_API_KEY

echo "=== [4/5] 서비스 계정 생성 및 최소 권한 부여 ==="
gcloud iam service-accounts create researchops-sa \
    --display-name="ResearchOps Agent Runtime SA" 2>/dev/null \
    || echo "이미 존재함 — 건너뜀"

for ROLE in \
    roles/secretmanager.secretAccessor \
    roles/logging.logWriter \
    roles/monitoring.metricWriter; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --condition=None 2>/dev/null || true
done

echo "=== [5/5] GitHub Actions Workload Identity Federation 설정 ==="
gcloud iam service-accounts create github-actions-sa \
    --display-name="GitHub Actions Deploy SA" 2>/dev/null \
    || echo "이미 존재함 — 건너뜀"

for ROLE in \
    roles/run.developer \
    roles/artifactregistry.writer \
    roles/iam.serviceAccountUser; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${DEPLOY_SA}" \
        --role="${ROLE}" \
        --condition=None 2>/dev/null || true
done

gcloud iam workload-identity-pools create github-pool \
    --location=global \
    --display-name="GitHub Actions Pool" 2>/dev/null \
    || echo "이미 존재함 — 건너뜀"

gcloud iam workload-identity-pools providers create-oidc github-provider \
    --location=global \
    --workload-identity-pool=github-pool \
    --display-name="GitHub OIDC Provider" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'" 2>/dev/null \
    || echo "이미 존재함 — 건너뜀"

WIF_POOL_ID=$(gcloud iam workload-identity-pools describe github-pool \
    --location=global --format='value(name)')

gcloud iam service-accounts add-iam-policy-binding "${DEPLOY_SA}" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/${WIF_POOL_ID}/attribute.repository/${GITHUB_REPO}" \
    2>/dev/null || true

WIF_PROVIDER=$(gcloud iam workload-identity-pools providers describe github-provider \
    --location=global \
    --workload-identity-pool=github-pool \
    --format='value(name)')

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/researchops/researchops-agent"

echo ""
echo "============================================================"
echo "완료! GitHub Repository Secrets에 다음 값을 등록하세요:"
echo "------------------------------------------------------------"
echo "WIF_PROVIDER  : ${WIF_PROVIDER}"
echo "WIF_SA_EMAIL  : ${DEPLOY_SA}"
echo "GCP_PROJECT_ID: ${PROJECT_ID}"
echo "GCP_REGION    : ${REGION}"
echo "============================================================"
echo ""
echo "다음 단계: 첫 Cloud Run 배포"
echo ""
echo "  gcloud auth configure-docker ${REGION}-docker.pkg.dev"
echo "  docker build -t ${IMAGE}:latest ."
echo "  docker push ${IMAGE}:latest"
echo "  gcloud run deploy researchops-agent \\"
echo "    --image=${IMAGE}:latest \\"
echo "    --platform=managed \\"
echo "    --region=${REGION} \\"
echo "    --no-cpu-throttling \\"
echo "    --min-instances=1 \\"
echo "    --max-instances=3 \\"
echo "    --memory=2Gi \\"
echo "    --cpu=1 \\"
echo "    --concurrency=10 \\"
echo "    --port=8080 \\"
echo "    --timeout=300 \\"
echo "    --set-secrets=GOOGLE_API_KEY=google-api-key:latest \\"
echo "    --set-env-vars=GOOGLE_CLOUD_PROJECT=${PROJECT_ID} \\"
echo "    --service-account=${SA_EMAIL} \\"
echo "    --allow-unauthenticated"
