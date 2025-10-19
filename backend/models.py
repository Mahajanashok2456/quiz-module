from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

class TopicRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=150, description="Topic for quiz generation (3-150 characters)")

class QuizQuestion(BaseModel):
    question_id: str
    question: str
    options: List[str]
    correct_answer: str

class QuizInDB(BaseModel):
    id: Optional[str] = None
    topic: str
    questions: List[QuizQuestion]
    created_at: datetime = datetime.utcnow()

class QuizResponse(BaseModel):
    quiz_id: str
    topic: str
    questions: List[QuizQuestion]

class UserAnswer(BaseModel):
    question_id: str = Field(..., min_length=1, description="Question identifier")
    user_answer_key: str = Field(..., min_length=1, max_length=1, pattern=r'^[A-D]$', description="Answer choice (A, B, C, or D only)")

class AnswerSubmission(BaseModel):
    quiz_id: str = Field(..., min_length=1, description="Quiz identifier")
    answers: List[UserAnswer] = Field(..., min_items=1, description="List of user answers")

class QuizResultInDB(BaseModel):
    id: Optional[str] = None
    quiz_id: str
    user_answers: List[str]
    score: int
    total_questions: int
    percentage: float
    submitted_at: datetime = datetime.utcnow()

class ScoreResponse(BaseModel):
    score: int
    total_questions: int
    percentage: float

# Helper function to convert MongoDB document to Pydantic model
def quiz_helper(quiz) -> dict:
    return {
        "id": str(quiz["_id"]),
        "topic": quiz["topic"],
        "questions": quiz["questions"],
        "created_at": quiz["created_at"]
    }

def result_helper(result) -> dict:
    return {
        "id": str(result["_id"]),
        "quiz_id": result["quiz_id"],
        "user_answers": result["user_answers"],
        "score": result["score"],
        "total_questions": result["total_questions"],
        "percentage": result["percentage"],
        "submitted_at": result["submitted_at"]
    }