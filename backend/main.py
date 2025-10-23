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
import httpx
import asyncio
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
    CodeSubmissionRequest,
    CodeExecutionResult,
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

async def generate_quiz_with_gemini(topic: str, question_type: str = "multiple_choice", difficulty: str = "medium") -> List[QuizQuestion]:
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

    # Generate different prompts based on question type
    if question_type == "multiple_choice":
        prompt = f"""
        You are a helpful quiz generation assistant.
        Generate exactly 10 multiple-choice questions on the topic of '{topic}'.
        The difficulty level should be '{difficulty}'.
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

    elif question_type == "coding":
        prompt = f"""
        You are a helpful coding quiz generation assistant.
        Generate exactly three coding problems on the topic of '{topic}': one easy, one medium, and one hard.
        Each problem_statement MUST be formatted using Markdown. Use H3 headers (###), bold text (**...**), and clear sections for Scenario, Input Format, Output Format, and Example.
        The function name MUST be provided in the 'function_name' field. Use 'solve' as the default function name.
        You MUST respond with ONLY a valid JSON object.

        Here is the required JSON format:
        {{
          "coding_questions": [
            {{
              "question_id": "cq1",
              "difficulty": "easy",
              "function_name": "solve",
              "problem_statement": "### Sum Calculator\\n\\n**Scenario:** You are given two integers, a and b. **Task:** Write a function that returns their sum.\\n\\n**Input Format:** Two integers, a and b.\\n\\n**Output Format:** A single integer representing the sum.\\n\\n**Example:**\\n\\n- **Input:** a = 2, b = 3\\n- **Output:** 5",
              "starter_code": "def solve(a, b):\\n  # Your code here\\n  return 0",
              "test_cases": [
                {{"input": "2 3", "expected_output": "5"}},
                {{"input": "-1 5", "expected_output": "4"}}
              ]
            }},
            {{
              "question_id": "cq2",
              "difficulty": "medium",
              "function_name": "solve",
              "problem_statement": "### Maximum Finder\\n\\n**Scenario:** You are given a list of numbers. **Task:** Write a function that finds the maximum number in the list.\\n\\n**Input Format:** A list of integers, nums.\\n\\n**Output Format:** A single integer representing the maximum value.\\n\\n**Example:**\\n\\n- **Input:** [1, 3, 2]\\n- **Output:** 3",
              "starter_code": "def solve(nums):\\n  # Your code here\\n  return 0",
              "test_cases": [
                {{"input": "[1, 3, 2]", "expected_output": "3"}},
                {{"input": "[-5, -1, -3]", "expected_output": "-1"}}
              ]
            }},
            {{
              "question_id": "cq3",
              "difficulty": "hard",
              "function_name": "solve",
              "problem_statement": "### Binary Search\\n\\n**Scenario:** You are given a sorted list of numbers and a target value. **Task:** Write a function that implements a binary search to find the index of the target.\\n\\n**Input Format:** A sorted list of integers, arr, and an integer, target.\\n\\n**Output Format:** An integer representing the index of the target, or -1 if not found.\\n\\n**Example:**\\n\\n- **Input:** arr = [1, 2, 3, 4, 5], target = 3\\n- **Output:** 2",
              "starter_code": "def solve(arr, target):\\n  # Your code here\\n  return -1",
              "test_cases": [
                {{"input": "[1, 2, 3, 4, 5] 3", "expected_output": "2"}},
                {{"input": "[10, 20, 30, 40] 25", "expected_output": "-1"}}
              ]
            }}
          ]
        }}
        """

    elif question_type == "long_answers":
        prompt = f"""
        You are a helpful quiz generation assistant.
        Generate exactly 10 long-answer questions on the topic of '{topic}'.
        The difficulty level should be '{difficulty}'.
        You MUST respond with ONLY a valid JSON object.
        These questions should require detailed explanations (2-3 paragraphs).

        Here is the required JSON format:
        {{
          "questions": [
            {{
              "question_id": "q1",
              "question": "Explain the process of photosynthesis in detail.",
              "expected_keywords": ["chlorophyll", "sunlight", "carbon dioxide", "glucose", "oxygen"]
            }},
            {{
              "question_id": "q2",
              "question": "Describe the main causes and effects of World War I.",
              "expected_keywords": ["assassination", "alliances", "trench warfare", "treaty"]
            }}
          ]
        }}
        """

    elif question_type == "fill_in_the_blanks":
        prompt = f"""
        You are a helpful quiz generation assistant.
        Generate exactly 10 fill-in-the-blank questions on the topic of '{topic}'.
        The difficulty level should be '{difficulty}'.
        You MUST respond with ONLY a valid JSON object.
        Each question should have one word or short phrase as the answer.

        Here is the required JSON format:
        {{
          "questions": [
            {{
              "question_id": "q1",
              "sentence_with_blank": "The capital of France is ________.",
              "correct_word": "Paris"
            }},
            {{
              "question_id": "q2",
              "sentence_with_blank": "Water freezes at ________ degrees Celsius.",
              "correct_word": "0"
            }}
          ]
        }}
        """

    else:
        # Default to multiple choice for unknown types
        prompt = f"""
        You are a helpful quiz generation assistant.
        Generate exactly 10 multiple-choice questions on the topic of '{topic}'.
        The difficulty level should be '{difficulty}'.
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

            response_text = response.text
            logger.info(f"Received response from AI (length: {len(response_text)})")

            # Attempt to find JSON within potential markdown fences
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                potential_json = response_text[json_start:json_end+1]
                try:
                    quiz_data = json.loads(potential_json)
                    logger.info(f"Successfully extracted and parsed JSON on attempt {attempt + 1}")

                    # Basic validation (check if 'questions' or 'coding_questions' key exists and is a list)
                    questions_key = "coding_questions" if question_type == "coding" else "questions"
                    if isinstance(quiz_data.get(questions_key), list):
                        # Validate the response structure
                        if not isinstance(quiz_data, dict) or questions_key not in quiz_data:
                            logger.error(f"Invalid response structure on attempt {attempt + 1}")
                            if attempt == 2:
                                raise HTTPException(status_code=500, detail="AI returned invalid quiz structure")
                            continue

                        questions_list = quiz_data.get(questions_key, [])
                        if len(questions_list) == 0:
                            logger.error(f"No questions in AI response on attempt {attempt + 1}")
                            if attempt == 2:
                                raise HTTPException(status_code=500, detail="AI returned no questions")
                            continue

                        # Convert to QuizQuestion objects based on question type
                        questions = []
                        for i, q_data in enumerate(questions_list):
                            try:
                                if question_type == "multiple_choice":
                                    question = QuizQuestion(
                                        question_id=f"q{i+1}",
                                        question=q_data["question"],
                                        options=q_data["options"],
                                        correct_answer=q_data["correct_answer"]
                                    )
                                elif question_type == "coding":
                                    # For coding questions, store the problem statement as question
                                    # and use options to store test cases
                                    test_cases = q_data.get("test_cases", [])
                                    test_cases_str = [f"Input: {tc.get('input', '')} -> Output: {tc.get('expected_output', '')}" for tc in test_cases[:4]]
                                    while len(test_cases_str) < 4:
                                        test_cases_str.append("")

                                    question = QuizQuestion(
                                        question_id=q_data["question_id"],
                                        question=q_data["problem_statement"],
                                        options=test_cases_str,
                                        correct_answer=q_data.get("starter_code", "No starter code"),
                                        starter_code=q_data.get("starter_code"),
                                        difficulty=q_data.get("difficulty"),
                                        function_name=q_data.get("function_name", "solve")
                                    )
                                elif question_type == "long_answers":
                                    # For long answers, store keywords as options
                                    keywords = q_data.get("expected_keywords", [])
                                    options = [f"Keyword: {kw}" for kw in keywords[:4]]
                                    while len(options) < 4:
                                        options.append("")

                                    question = QuizQuestion(
                                        question_id=f"q{i+1}",
                                        question=q_data["question"],
                                        options=options,
                                        correct_answer="Detailed explanation required"
                                    )
                                elif question_type == "fill_in_the_blanks":
                                    # For fill in blanks, create options with the correct answer and distractors
                                    correct_word = q_data.get("correct_word", "")
                                    options = [
                                        f"A) {correct_word}",
                                        f"B) Incorrect",
                                        f"C) Wrong",
                                        f"D) Not correct"
                                    ]

                                    question = QuizQuestion(
                                        question_id=f"q{i+1}",
                                        question=q_data["sentence_with_blank"],
                                        options=options,
                                        correct_answer="A"
                                    )
                                else:
                                    # Default fallback
                                    question = QuizQuestion(
                                        question_id=f"q{i+1}",
                                        question=str(q_data),
                                        options=["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
                                        correct_answer="A"
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
                    else:
                        logger.error(f"Extracted JSON missing 'questions' list on attempt {attempt + 1}. JSON: {potential_json[:200]}...")
                        if attempt == 2:
                            raise HTTPException(status_code=500, detail=f"AI service error: {response_text[:200]}...")
                        continue
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing failed on attempt {attempt + 1}: {json_err}")
                    logger.debug(f"Clean response text (first 1000 chars): {potential_json[:1000]}")
                    logger.debug(f"Clean response text (length: {len(potential_json)})")
                    if attempt == 2:
                        raise HTTPException(status_code=500, detail=f"AI service error: {response_text[:200]}...")
                    continue
            else:
                logger.error(f"Could not find valid JSON boundaries in response on attempt {attempt + 1}. Response: {response_text[:200]}...")
                if attempt == 2:
                    raise HTTPException(status_code=500, detail=f"AI service error: {response_text[:200]}...")
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
        questions = await generate_quiz_with_gemini(topic_request.topic, topic_request.question_type or "multiple_choice", topic_request.difficulty or "medium")
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

        if topic_request.question_type == "coding":
            response = QuizResponse(
                quiz_id=quiz_id,
                topic=topic_request.topic,
                questions=[],
                coding_questions=sanitized_questions
            )
        else:
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

# Code execution endpoint
@app.post("/api/quiz/run_code", response_model=CodeExecutionResult)
@limiter.limit("10/minute")
async def run_code(submission: CodeSubmissionRequest, request: Request):
    """Execute user code against test cases for a coding question"""
    try:
        # Retrieve the quiz
        quiz = await get_quiz_from_db(submission.quiz_id)
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        # Find the specific coding question
        question = next((q for q in quiz["questions"] if q.get("question_id") == submission.question_id), None)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Get test cases (stored as strings in options)
        test_cases_str = question.get("options", [])
        test_cases = []
        for tc_str in test_cases_str:
            # Parse the string format: "Input: ... -> Output: ..."
            if "Input:" in tc_str and "Output:" in tc_str:
                parts = tc_str.split(" -> ")
                if len(parts) == 2:
                    input_part = parts[0].replace("Input: ", "").strip()
                    output_part = parts[1].replace("Output: ", "").strip()
                    test_cases.append({"input": input_part, "expected_output": output_part})

        if not test_cases:
            raise HTTPException(status_code=500, detail="No test cases found for this question.")

        # --- START JUDGE0 INTEGRATION ---
        judge0_key = os.getenv("RAPIDAPI_JUDGE0_KEY")
        judge0_host = os.getenv("RAPIDAPI_JUDGE0_HOST")

        print(f"DEBUG: Judge0 Key: {judge0_key}")
        print(f"DEBUG: Judge0 Host: {judge0_host}")

        if not judge0_key or not judge0_host:
            logging.error("Judge0 API Key or Host not found in environment variables.")
            raise HTTPException(status_code=500, detail="Code execution service is not configured.")

        # Map language names to Judge0 language IDs
        language_map = {
            "python": 71,
            "java": 62,
            "cpp": 54,  # For C++ (GCC 9.2.0)
        }
        language_id = language_map.get(submission.language.lower())
        if not language_id:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {submission.language}")

        passed_all_tests = True
        combined_output = ""
        error_output = None

        # Run code against each test case
        async with httpx.AsyncClient() as client:
            for i, case in enumerate(test_cases):
                input_data = case.get("input")
                expected_output_data = case.get("expected_output", "").strip()

                logging.debug(f"Running test case {i+1}: Input='{input_data}'")

                # Build full script for execution
                func_name = question.get("function_name", "solve")

                if submission.language.lower() == "python":
                    # For Python, combine user code with test execution logic
                    full_code = f"""
{submission.user_code}

# Test case execution logic
import sys
input_value = sys.stdin.read().strip()
input_parts = input_value.split()
if len(input_parts) == 2:
    a = int(input_parts[0])
    b = int(input_parts[1])
    result = {func_name}(a, b)
    print(result)
else:
    # For single input
    result = {func_name}(input_value)
    print(result)
"""
                else:
                    # For other languages, use the user code as is (simplified)
                    full_code = submission.user_code

                submission_payload = {
                    "language_id": language_id,
                    "source_code": full_code,
                    "stdin": input_data
                }
                headers = {
                    "content-type": "application/json",
                    "X-RapidAPI-Key": judge0_key,
                    "X-RapidAPI-Host": judge0_host
                }
                create_url = "https://judge0-ce.p.rapidapi.com/submissions?base64_encoded=false&wait=false"

                try:
                    # 1. Create Submission
                    print(f"DEBUG: Creating submission for test case {i+1}")
                    print(f"DEBUG: Payload: {submission_payload}")
                    print(f"DEBUG: Headers: {headers}")
                    post_response = await client.post(create_url, headers=headers, json=submission_payload, timeout=30.0)
                    print(f"DEBUG: Post response status: {post_response.status_code}")
                    print(f"DEBUG: Post response text: {post_response.text}")
                    post_response.raise_for_status()
                    submission_data = post_response.json()
                    token = submission_data.get("token")
                    print(f"DEBUG: Submission token: {token}")
                    if not token:
                        raise HTTPException(status_code=500, detail="Failed to get submission token from Judge0.")

                    # 2. Poll for Result
                    get_url = f"https://judge0-ce.p.rapidapi.com/submissions/{token}?base64_encoded=false&fields=status_id,stdout,stderr,compile_output,message,time,memory"
                    status_id = 1  # Processing

                    while status_id <= 2:  # While Queued or Processing
                        await asyncio.sleep(1)  # Wait 1 second before polling again
                        get_response = await client.get(get_url, headers=headers, timeout=30.0)
                        print(f"DEBUG: Get response status: {get_response.status_code}")
                        print(f"DEBUG: Get response text: {get_response.text}")
                        get_response.raise_for_status()
                        result_data = get_response.json()
                        status_id = result_data.get("status_id", 0)
                        logging.debug(f"Polling submission {token}: Status ID = {status_id}")

                    # 3. Process Result
                    stdout = (result_data.get("stdout") or "").strip()
                    stderr = result_data.get("stderr")
                    compile_output = result_data.get("compile_output")
                    message = result_data.get("message")

                    print(f"DEBUG: Test case {i+1} result - Status: {status_id}, Message: {message}")
                    print(f"DEBUG: Stdout: {stdout}")
                    print(f"DEBUG: Stderr: {stderr}")
                    print(f"DEBUG: Compile output: {compile_output}")

                    # --- BEGIN NEW MARKDOWN FORMATTING ---
                    # Determine the execution status message
                    if status_id == 3:
                        status_message = "✅ Accepted (Passed)"
                    elif status_id == 4:
                        status_message = "❌ Wrong Answer"
                    elif status_id == 6:
                        status_message = "❌ Compilation Error"
                    else:
                        status_message = f"🛑 Failed (Status ID: {status_id})"

                    # Start Markdown for this test case
                    case_output_markdown = f"### Test Case {i+1}: {status_message}\n"

                    # Add Inputs/Outputs using Markdown code blocks for clarity
                    case_output_markdown += "| Detail | Expected | Your Output |\n"
                    case_output_markdown += "| :--- | :--- | :--- |\n"
                    case_output_markdown += f"| **Input** | `{input_data}` | N/A |\n"
                    case_output_markdown += f"| **Output** | `{expected_output_data}` | `{stdout}` |\n"

                    # Add detailed error messages below the table if present
                    if stderr:
                        case_output_markdown += f"\n**Runtime Error (stderr):**\n```\n{stderr}\n```\n"
                        error_output = (error_output or "") + f"Test {i+1} Runtime Error: {stderr[:50]}...\n"
                    if compile_output:
                        case_output_markdown += f"\n**Compilation Error:**\n```\n{compile_output}\n```\n"
                        error_output = (error_output or "") + f"Test {i+1} Compile Error: {compile_output[:50]}...\n"

                    case_output_markdown += "\n---\n"

                    combined_output += case_output_markdown
                    # --- END NEW MARKDOWN FORMATTING ---

                    # Check if the test case passed
                    is_output_correct = stdout.strip() == expected_output_data.strip()
                    is_test_case_passed = (status_id == 3) and is_output_correct

                    if not is_test_case_passed:
                        passed_all_tests = False
                        logging.warning(f"Test case {i+1} failed. Status: {status_id}, Output Match: {is_output_correct}")

                except httpx.HTTPStatusError as exc:
                    logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
                    print(f"DEBUG: HTTP error: {exc.response.status_code} - {exc.response.text}")
                    raise HTTPException(status_code=500, detail=f"Error communicating with code execution service: Status {exc.response.status_code}")
                except httpx.RequestError as exc:
                    logging.error(f"Request error occurred: {exc}")
                    print(f"DEBUG: Request error: {exc}")
                    raise HTTPException(status_code=500, detail="Error connecting to code execution service.")
                except Exception as e:
                    logging.error(f"An unexpected error occurred during code execution: {e}")
                    print(f"DEBUG: Unexpected error: {e}")
                    raise HTTPException(status_code=500, detail=f"Unexpected code execution error: {str(e)}")

        # Prepare final result
        result = CodeExecutionResult(
            passed=passed_all_tests,
            output=combined_output.strip(),
            error=error_output
        )
        # --- END JUDGE0 INTEGRATION ---

        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error running code: {e}")
        raise HTTPException(status_code=500, detail=f"Error running code: {str(e)}")

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