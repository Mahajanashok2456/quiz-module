from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId
import json
import logging
import google.generativeai as genai
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Import our models
from models import (
    TopicRequest,
    QuizQuestion,
    QuizResponse,
    AnswerSubmission,
    UserAnswer,
    ScoreResponse,
    QuizInDB,
    QuizResultInDB,
    quiz_helper,
    result_helper
)

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI(
    title="AI Quiz Generator API",
    description="Backend API for AI-powered quiz generation using Google Gemini",
    version="1.0.0"
)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Set up production CORS
origins = [
    "http://localhost:3000",  # For development
    "https://www.my-production-domain.com" # <-- REPLACE THIS WITH YOUR ACTUAL PRODUCTION DOMAIN
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# MongoDB client
mongodb_client = None
database = None

@app.on_event("startup")
async def startup_db_client():
    """Initialize MongoDB connection on startup"""
    global mongodb_client, database
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "ai_quiz_generator")

    mongodb_client = AsyncIOMotorClient(mongodb_url)
    database = mongodb_client[database_name]

    # Test the connection
    try:
        await mongodb_client.admin.command('ping')
        print(f"Connected to MongoDB: {mongodb_url}")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    """Close MongoDB connection on shutdown"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()

# Database service functions
async def save_quiz_to_db(topic: str, questions: List[QuizQuestion]):
    """Save a complete quiz to MongoDB"""
    try:
        print(f"DEBUG: Saving quiz to DB - Topic: {topic}")
        print(f"DEBUG: Number of questions: {len(questions)}")

        # Check if database is available
        if database is None:
            raise Exception("Database connection is None")

        print(f"DEBUG: Database collection: {database.quizzes}")
        print(f"DEBUG: Database name: {database.name}")

        quiz_data = {
            "topic": topic,
            "questions": [q.model_dump() for q in questions],  # Pydantic v2 uses model_dump()
            "created_at": datetime.utcnow()
        }
        print(f"DEBUG: Quiz data prepared, inserting...")
        print(f"DEBUG: Quiz data keys: {list(quiz_data.keys())}")
        print(f"DEBUG: Questions count in data: {len(quiz_data['questions'])}")

        result = await database.quizzes.insert_one(quiz_data)
        print(f"DEBUG: Quiz inserted successfully with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        print(f"DEBUG: Error saving quiz to database: {e}")
        print(f"DEBUG: Error type: {type(e)}")
        import traceback
        print(f"DEBUG: Database error traceback: {traceback.format_exc()}")
        raise

async def get_quiz_from_db(quiz_id: str):
    """Retrieve a quiz from MongoDB by ID"""
    try:
        quiz = await database.quizzes.find_one({"_id": ObjectId(quiz_id)})
        if quiz:
            return quiz_helper(quiz)
    except Exception as e:
        print(f"Error retrieving quiz: {e}")
    return None

async def save_quiz_result(quiz_id: str, user_answers: List[str], score: int, total: int):
    """Save quiz result to MongoDB"""
    percentage = (score / total) * 100 if total > 0 else 0
    result_data = {
        "quiz_id": quiz_id,
        "user_answers": user_answers,
        "score": score,
        "total_questions": total,
        "percentage": percentage,
        "submitted_at": datetime.utcnow()
    }
    result = await database.results.insert_one(result_data)
    return str(result.inserted_id)

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

async def generate_quiz_with_gemini(topic: str) -> List[QuizQuestion]:
    """Generate quiz questions using Google Gemini AI with retry logic"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    # Configure logging
    logger = logging.getLogger(__name__)
    logger.info(f"Generating quiz for topic: '{topic}' using Gemini API")
    logger.info(f"Gemini API key configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"Gemini API key length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0}")
    logger.info(f"Gemini API key preview: {GEMINI_API_KEY[:20]}..." if GEMINI_API_KEY and len(GEMINI_API_KEY) > 20 else "API key too short or not set")

    # Validate API key format (Gemini keys typically start with "AIza")
    if GEMINI_API_KEY and not GEMINI_API_KEY.startswith("AIza"):
        logger.warning("API key doesn't start with 'AIza' - may be invalid Gemini API key format")

    # Try different model names in case the primary one doesn't exist
    model_names = ['gemini-2.5-flash', 'gemini-pro', 'gemini-1.5-pro', 'gemini-1.0-pro']

    model = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            logger.info(f"Successfully initialized model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to initialize model {model_name}: {e}")
            continue

    if model is None:
        # Fallback: Return a mock quiz for testing if API key is invalid
        logger.warning("No Gemini models available - returning mock quiz for testing")
        return [
            QuizQuestion(
                question_id="q1",
                question="What is the capital of France?",
                options=["A) London", "B) Berlin", "C) Paris", "D) Madrid"],
                correct_answer="C"
            ),
            QuizQuestion(
                question_id="q2",
                question="What is 2 + 2?",
                options=["A) 3", "B) 4", "C) 5", "D) 6"],
                correct_answer="B"
            ),
            QuizQuestion(
                question_id="q3",
                question="Which planet is known as the Red Planet?",
                options=["A) Venus", "B) Mars", "C) Jupiter", "D) Saturn"],
                correct_answer="B"
            ),
            QuizQuestion(
                question_id="q4",
                question="What does 'AI' stand for?",
                options=["A) Artificial Intelligence", "B) Automated Interface", "C) Active Internet", "D) Automatic Integration"],
                correct_answer="A"
            ),
            QuizQuestion(
                question_id="q5",
                question="In which year was Python first released?",
                options=["A) 1989", "B) 1991", "C) 1995", "D) 2000"],
                correct_answer="B"
            )
        ]

    prompt = f"""
    You are a helpful quiz generation assistant.
    Generate 5 multiple-choice questions on the topic of '{topic}'.
    You MUST respond with ONLY a valid JSON object.
    The JSON object must follow this exact format.
    The "options" must be an array of strings.
    The "correct_answer" MUST be ONLY the capital letter (A, B, C, or D) that corresponds to the correct option.

    Here is the required JSON format:
    {{
      "questions": [
        {{
          "question_id": "q1",
          "question": "What is the capital of France?",
          "options": [
            "A) London",
            "B) Berlin",
            "C) Paris",
            "D) Madrid"
          ],
          "correct_answer": "C"
        }},
        {{
          "question_id": "q2",
          "question": "What is 2+2?",
          "options": [
            "A) 3",
            "B) 4",
            "C) 5",
            "D) 6"
          ],
          "correct_answer": "B"
        }}
      ]
    }}
    """

    # Retry loop with 3 attempts
    for attempt in range(3):
        try:
            logger.info(f"Attempting to generate quiz for topic '{topic}' (attempt {attempt + 1}/3)")

            # Generate content from AI
            response = model.generate_content(prompt)

            # Check if response has text content
            if not response or not response.text:
                logger.warning(f"Empty response from AI on attempt {attempt + 1}")
                if attempt == 2:  # Last attempt
                    raise HTTPException(status_code=500, detail="AI returned empty response after 3 attempts")
                continue

            response_text = response.text.strip()
            logger.info(f"Received response from AI (length: {len(response_text)})")

            # Handle non-JSON responses (like error messages)
            if not (response_text.startswith('{') and response_text.endswith('}')):
                logger.error(f"AI returned non-JSON response on attempt {attempt + 1}")
                logger.error(f"Response preview: {response_text[:500]}")
                logger.error(f"Response ends with: '{response_text[-100:]}'")

                # Try to extract JSON if it looks like it contains JSON
                if '{' in response_text and '}' in response_text:
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    potential_json = response_text[start_idx:end_idx]
                    logger.error(f"Attempting to extract JSON: {potential_json[:300]}")

                    # Try parsing the extracted JSON
                    try:
                        test_data = json.loads(potential_json)
                        logger.info("Successfully extracted and parsed JSON from response")
                        clean_response_text = potential_json
                    except json.JSONDecodeError:
                        logger.error("Failed to extract valid JSON from response")
                        if attempt == 2:
                            raise HTTPException(status_code=500, detail=f"AI service error: {response_text[:200]}...")
                        continue
                else:
                    if attempt == 2:
                        raise HTTPException(status_code=500, detail=f"AI service error: {response_text[:200]}...")
                    continue

            # Extract JSON from markdown code blocks if present
            clean_response_text = response_text
            if '```json' in response_text:
                # Handle ```json format
                parts = response_text.split('```json')
                if len(parts) > 1:
                    json_part = parts[1]
                    if '```' in json_part:
                        clean_response_text = json_part.split('```')[0].strip()
                    else:
                        clean_response_text = json_part.strip()
            elif '```' in response_text:
                # Handle generic ``` format
                parts = response_text.split('```')
                if len(parts) >= 3:  # Should have at least ``` content ```
                    clean_response_text = parts[1].strip()

            # Parse the JSON response
            try:
                quiz_data = json.loads(clean_response_text)
                logger.info("Successfully parsed JSON response from AI")

                # Validate the response structure
                if not isinstance(quiz_data, dict) or "questions" not in quiz_data:
                    logger.error(f"Invalid response structure on attempt {attempt + 1}")
                    if attempt == 2:
                        raise HTTPException(status_code=500, detail="AI returned invalid quiz structure")
                    continue

                questions_list = quiz_data.get("questions", [])
                if len(questions_list) == 0:
                    logger.error(f"No questions in AI response on attempt {attempt + 1}")
                    if attempt == 2:
                        raise HTTPException(status_code=500, detail="AI returned no questions")
                    continue

                # Convert to QuizQuestion objects
                questions = []
                for i, q_data in enumerate(questions_list):
                    try:
                        question = QuizQuestion(
                            question_id=f"q{i+1}",
                            question=q_data["question"],
                            options=q_data["options"],
                            correct_answer=q_data["correct_answer"]
                        )
                        questions.append(question)
                    except (KeyError, TypeError) as e:
                        logger.error(f"Invalid question data on attempt {attempt + 1}: {e}")
                        if attempt == 2:
                            raise HTTPException(status_code=500, detail=f"AI returned malformed question data: {str(e)}")
                        break

                if len(questions) == 0:
                    logger.error(f"No valid questions could be created on attempt {attempt + 1}")
                    if attempt == 2:
                        raise HTTPException(status_code=500, detail="Could not create any valid questions")
                    continue

                logger.info(f"Successfully generated {len(questions)} questions on attempt {attempt + 1}")
                return questions

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                logger.error(f"Clean response text (first 1000 chars): {clean_response_text[:1000]}")
                logger.error(f"Clean response text (length: {len(clean_response_text)})")
                if attempt == 2:  # Last attempt
                    raise HTTPException(status_code=500, detail=f"AI failed to generate valid JSON after 3 attempts. Response: {clean_response_text[:200]}...")
                continue

        except Exception as e:
            logger.error(f"Error calling Gemini API on attempt {attempt + 1}: {e}")
            if attempt == 2:  # Last attempt
                raise HTTPException(status_code=500, detail=f"AI service error after 3 attempts: {str(e)}")

    # This should never be reached, but just in case
    raise HTTPException(status_code=500, detail="AI failed to generate a valid quiz. Please try again.")

# Pydantic models are now imported from models.py

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint for API health check"""
    return {"message": "AI Quiz Generator API is running!"}

# Quiz generation endpoint
@app.post("/api/quiz/generate", response_model=QuizResponse)
@limiter.limit("10/minute")
async def generate_quiz(topic_request: TopicRequest, request: Request):
    """Generate a quiz for the given topic using Google Gemini AI"""
    try:
        print(f"DEBUG: Starting quiz generation for topic: '{topic_request.topic}'")
        print(f"DEBUG: Topic length: {len(topic_request.topic)}")
        print(f"DEBUG: Topic content: {topic_request.topic}")

        # Generate quiz using Gemini API
        print("DEBUG: Calling generate_quiz_with_gemini...")
        questions = await generate_quiz_with_gemini(topic_request.topic)
        print(f"DEBUG: Generated {len(questions)} questions")

        # Save complete quiz to database
        print("DEBUG: Saving quiz to database...")
        print(f"DEBUG: Database object: {database}")
        print(f"DEBUG: Database type: {type(database)}")
        quiz_id = await save_quiz_to_db(topic_request.topic, questions)
        print(f"DEBUG: Quiz saved with ID: {quiz_id}")

        # Return sanitized quiz (without correct answers) for frontend
        print("DEBUG: Creating sanitized response...")
        sanitized_questions = []
        for q in questions:
            sanitized_questions.append(QuizQuestion(
                question_id=q.question_id,
                question=q.question,
                options=q.options,
                correct_answer=""  # Don't send correct answer to frontend
            ))

        response = QuizResponse(
            quiz_id=quiz_id,
            topic=topic_request.topic,
            questions=sanitized_questions
        )
        print(f"DEBUG: Returning response with {len(sanitized_questions)} questions")
        return response

    except Exception as e:
        print(f"DEBUG: Exception in generate_quiz: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

# Quiz submission endpoint
@app.post("/api/quiz/submit", response_model=ScoreResponse)
async def submit_quiz(submission: AnswerSubmission):
    """Submit quiz answers and get score"""
    try:
        # Validate and convert quiz_id to ObjectId
        try:
            quiz_object_id = ObjectId(submission.quiz_id)
        except InvalidId:
            raise HTTPException(
                status_code=400,
                detail="Invalid quiz_id format"
            )

        # Retrieve the correct answers from database
        print(f"DEBUG: Retrieving quiz {submission.quiz_id} from database")
        quiz = await get_quiz_from_db(submission.quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        print(f"DEBUG: Quiz retrieved successfully")
        print(f"DEBUG: Quiz questions count: {len(quiz.get('questions', []))}")
        if quiz.get('questions'):
            print(f"DEBUG: First question keys: {list(quiz['questions'][0].keys()) if quiz['questions'] else 'No questions'}")
            print(f"DEBUG: First question data: {quiz['questions'][0] if quiz['questions'] else 'No questions'}")

        # Calculate score
        score = 0
        print(f"DEBUG: Processing {len(submission.answers)} user answers")
        print(f"DEBUG: Quiz has {len(quiz['questions'])} questions in DB")

        for user_ans in submission.answers:
            print(f"DEBUG: Looking for question_id: {user_ans.question_id}")

            # Find the matching question from the quiz in the DB
            question_from_db = next((q for q in quiz["questions"] if q.get("question_id") == user_ans.question_id), None)

            if question_from_db:
                user_answer_str = user_ans.user_answer_key
                correct_answer_str = question_from_db.get('correct_answer', 'NOT_FOUND')

                print(f"DEBUG: Question found in DB")
                print(f"DEBUG: Question keys: {list(question_from_db.keys())}")
                print(f"DEBUG: User's Answer: {user_answer_str}")
                print(f"DEBUG: Correct Answer: {correct_answer_str}")
                print(f"DEBUG: Match: {user_answer_str == correct_answer_str}")
                print("-------------------------")

                if user_answer_str == correct_answer_str:
                    score += 1
            else:
                print(f"DEBUG: Could not find question {user_ans.question_id} in database quiz")
                print(f"DEBUG: Available question IDs: {[q.get('question_id') for q in quiz['questions']]}")

        total_questions = len(submission.answers)

        # Save the result (convert UserAnswer objects to strings for storage)
        user_answers_str = [ua.user_answer_key for ua in submission.answers]
        await save_quiz_result(
            submission.quiz_id,
            user_answers_str,
            score,
            total_questions
        )

        percentage = (score / total_questions) * 100 if total_questions > 0 else 0

        return ScoreResponse(
            score=score,
            total_questions=total_questions,
            percentage=round(percentage, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in submit_quiz: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )