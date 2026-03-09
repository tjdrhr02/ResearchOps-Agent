from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_note_service
from src.api.schemas.request import SaveNoteRequest
from src.api.schemas.response import SaveNoteResponse, SearchNotesResponse
from src.application.services.note_service import NoteService

router = APIRouter()


@router.post("/save", response_model=SaveNoteResponse)
async def save_note(
    request: SaveNoteRequest,
    note_service: NoteService = Depends(get_note_service),
) -> SaveNoteResponse:
    note_id = await note_service.save(request.note)
    return SaveNoteResponse(note_id=note_id)


@router.get("/search", response_model=SearchNotesResponse)
async def search_notes(
    keyword: str = Query(default="", min_length=0),
    note_service: NoteService = Depends(get_note_service),
) -> SearchNotesResponse:
    items = await note_service.search(keyword=keyword)
    return SearchNotesResponse(items=items)
