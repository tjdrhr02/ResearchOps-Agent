#!/usr/bin/env bash
# ResearchOps Agent — GCP 프로젝트 초기 설정 스크립트
# 실행 전: PROJECT_ID, GITHUB_REPO 변수 수정 필수
set -euo pipefail

# ── 사용자 설정 변수 ──────────────────────────────────────────────
PROJECT_ID="researchops-agent-$(date +%Y%m)"       # 예: researchops-agent-202604
REGION="asia-northeast3"                            # 서울 리전
GITHUB_REPO="FTinMacBook/ResearchOps-Agent"        # GitHub owner/repo
BILLING_ACCOUNT=$(gcloud billing accounts list --format='value(name)' --limit=1)
# ─────────────────────────────────────────────────────────────────

echo "=== [1/6] 프로젝트 생성 및 활성화 ==="
gcloud projects create "${PROJECT_ID}" --name="ResearchOps Agent"
gcloud config set project "${PROJECT_ID}"
gcloud billing projects link "${PROJECT_ID}" --billing-account="${BILLING_ACCOUNT}"

echo "=== [2/6] 필요한 API 활성화 ==="
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    iam.googleapis.com \
    iamcredentials.googleapis.com

echo "=== [3/6] Artifact Registry 저장소 생성 ==="
gcloud artifacts repositories create researchops \
    --repository-format=docker \
    --location="${REGION}" \
    --description="ResearchOps Agent container images"

echo "=== [4/6] Secret Manager에 GOOGLE_API_KEY 등록 ==="
# ⚠️ 주의: 아래 명령 실행 전 실제 API 키 입력
read -rsp "GOOGLE_API_KEY 입력: " GOOGLE_API_KEY
echo ""
echo -n "${GOOGLE_API_KEY}" | gcloud secrets create google-api-key \
    --data-file=- \
    --replication-policy=automatic
unset GOOGLE_API_KEY

echo "=== [5/6] 서비스 계정 생성 및 최소 권한 부여 ==="
SA_EMAIL="researchops-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create researchops-sa \
    --display-name="ResearchOps Agent Runtime SA"

for ROLE in \
    roles/secretmanager.secretAccessor \
    roles/logging.logWriter \
    roles/monitoring.metricWriter; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}"
done

echo "=== [6/6] GitHub Actions Workload Identity Federation 설정 ==="
DEPLOY_SA="github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create github-actions-sa \
    --display-name="GitHub Actions Deploy SA"

for ROLE in \
    roles/run.developer \
    roles/artifactregistry.writer \
    roles/iam.serviceAccountUser; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member="serviceAccount:${DEPLOY_SA}" \
        --role="${ROLE}"
done

gcloud iam workload-identity-pools create github-pool \
    --location=global \
    --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc github-provider \
    --location=global \
    --workload-identity-pool=github-pool \
    --display-name="GitHub OIDC Provider" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='${GITHUB_REPO}'"

WIF_POOL_ID=$(gcloud iam workload-identity-pools describe github-pool \
    --location=global --format='value(name)')

gcloud iam service-accounts add-iam-policy-binding "${DEPLOY_SA}" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/${WIF_POOL_ID}/attribute.repository/${GITHUB_REPO}"

WIF_PROVIDER=$(gcloud iam workload-identity-pools providers describe github-provider \
    --location=global \
    --workload-identity-pool=github-pool \
    --format='value(name)')

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
echo "다음 단계: 첫 Cloud Run 배포 (아래 명령 실행)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/researchops/researchops-agent"
echo ""
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
