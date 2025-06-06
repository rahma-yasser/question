import logging
import json
import re
import uuid
from typing import List, Optional
from datetime import datetime
from contextlib import contextmanager
import google.generativeai as genai
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fastapi import FastAPI, HTTPException, Query
import asyncio
import os
from dotenv import load_dotenv

# Logging configuration with request ID support
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [RequestID: %(request_id)s] - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@contextmanager
def log_context(request_id: str):
    """Add request ID to logging context."""
    logging.getLogger().handlers[0].setFormatter(
        logging.Formatter(f"%(asctime)s - %(levelname)s - [RequestID: {request_id}] - %(message)s")
    )
    try:
        yield
    finally:
        logging.getLogger().handlers[0].setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - [RequestID: None] - %(message)s")
        )

# FastAPI app
app = FastAPI(
    title="Question Generation API",
    description="API for generating interview questions and answers using Gemini models.",
    version="1.0.3"
)

# Track configurations
TRACKS = {
    "1": {
        "name": "flutter developer",
        "default_topic": "flutter developer",
        "tuned_model": "tunedModels/fluttermodel-2cx3qf2cm72f"
    },
    "2": {
        "name": "machine learning",
        "default_topic": "machine learning",
        "tuned_model": "tunedModels/chk1-607sqy6pv5wt"
    },
    "3": {
        "name": "backend.net",
        "default_topic": "backend.net",
        "tuned_model": "gemini-1.5-flash"
    },
    "4": {
        "name": "frontend",
        "default_topic": "frontend",
        "tuned_model": "gemini-1.5-flash"
    }
}

# Pydantic models
class Question(BaseModel):
    question: str
    gemini_answer: str
    user_answer: str
    topic: str
    classification: str

class TopicQuestions(BaseModel):
    topic: str
    questions: List[Question]

class QuestionResponse(BaseModel):
    topics: List[TopicQuestions]

class GenerateQuestionsRequest(BaseModel):
    track_id: Optional[str] = Field(None, description="Track ID (e.g., '1' for Flutter, '2' for Machine Learning, '3' for Backend.NET, '4' for Frontend)")
    topics: Optional[List[str]] = Field(None, description="List of custom topics (e.g., ['pandas', 'numpy', 'asp.net', 'react'])")
    difficulty: str = Field("beginner", description="Difficulty level: beginner, intermediate, or advanced")
    num_questions: int = Field(3, ge=1, le=20, description="Number of questions (1 to 20)")

# Question Generator class
class QuestionGenerator:
    """A system to generate questions and answers using Google Gemini API."""
    
    def __init__(self):
        """Initialize the question generator."""
        self.google_api_key = None
        self.question_model = None
        self.rate_limit_delay = 2

    def setup_environment(self) -> None:
        """Load environment variables and configure Google API."""
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            logger.error("GOOGLE_API_KEY environment variable is not set")
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=self.google_api_key)
        logger.info("Environment setup completed.")

    def initialize_model(self, model_name: str = "gemini-1.5-flash") -> None:
        """Initialize the specified Gemini model."""
        try:
            self.question_model = genai.GenerativeModel(model_name)
            logger.info(f"Model '{model_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize model '{model_name}': {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        after=lambda retry_state: logger.info(f"Retrying content generation, attempt {retry_state.attempt_number}")
    )
    async def generate_content(self, prompt: str, response_type: str = "text/plain") -> str:
        """Generate content using the Gemini API with retry on rate limits."""
        if not self.question_model:
            logger.error("Model not initialized")
            raise ValueError("Model not initialized")
        try:
            async with asyncio.timeout(10):
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.question_model.generate_content(
                        prompt,
                        generation_config={
                            "response_mime_type": response_type,
                            "temperature": 0.8
                        }
                    )
                )
                await asyncio.sleep(self.rate_limit_delay)
                return response.text
        except asyncio.TimeoutError:
            logger.error("Gemini API call timed out")
            raise HTTPException(status_code=504, detail="Gemini API request timed out")
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise

# Helper function to generate questions for a topic
async def generate_questions_for_topic(
    generator: QuestionGenerator,
    topic: str,
    track_id: Optional[str] = None,
    difficulty: str = "beginner",
    num_questions: int = 3
) -> TopicQuestions:
    """Generate questions and answers for a single topic using Gemini."""
    request_id = str(uuid.uuid4())
    with log_context(request_id):
        selected_topic = topic.lower().strip()
        if not selected_topic:
            logger.error("Topic cannot be empty.")
            raise HTTPException(status_code=400, detail="Topic cannot be empty.")

        question_list = []

        try:
            # Select model based on track_id
            model_name = TRACKS.get(track_id, {}).get("tuned_model", "gemini-1.5-flash")
            generator.initialize_model(model_name)

            question_prompt = (
                f"Generate a JSON array of {num_questions} {difficulty}-level questions about {selected_topic}, "
                f"each with a 'question' field and a 'gemini_answer' field. Ensure questions are varied and unique, "
                f"avoiding repetition of common questions. For each answer: "
                f"- Base the answer on the {difficulty} skill level. "
                f"- Make the answer clear, concise, and {difficulty}-friendly. "
                f"- Include a simple explanation of the concept. "
                f"- Avoid unnecessary repetition or overly complex terms. "
                f'Example: [{{"question": "What is a key feature of {selected_topic}?", "gemini_answer": "A simple explanation here."}}]'
            )
            response_text = await generator.generate_content(
                question_prompt,
                response_type="application/json"
            )

            try:
                questions_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse API response as JSON: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to parse API response as JSON")

            if not isinstance(questions_data, list):
                logger.error(f"API response is not a list of questions")
                raise HTTPException(status_code=500, detail="API response is not a list of questions")
            if not questions_data:
                logger.warning(f"No questions returned for topic '{selected_topic}'")
                return TopicQuestions(topic=selected_topic, questions=[])

            if len(questions_data) < num_questions:
                logger.warning(f"Expected {num_questions} questions, got {len(questions_data)}")

            for q in questions_data:
                if not isinstance(q, dict) or not q.get("question") or not q.get("gemini_answer") or not isinstance(q.get("question"), str) or len(q["question"].strip()) == 0:
                    logger.warning(f"Skipping invalid question: {q.get('question', 'None')}")
                    continue

                question_list.append(Question(
                    question=q.get("question", "").strip(),
                    gemini_answer=re.sub(r'[^\x00-\x7F]+', '', re.sub(r'\s+', ' ', q.get("gemini_answer", "").strip())),
                    user_answer="",
                    topic=selected_topic,
                    classification=difficulty.capitalize()
                ))

            return TopicQuestions(topic=selected_topic, questions=question_list)

        except Exception as e:
            logger.error(f"Error generating questions for topic '{selected_topic}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

# Startup event to validate configuration
@app.on_event("startup")
async def startup_event():
    """Validate environment variables and TRACKS configuration on startup."""
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY environment variable is not set")
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
    
    for track_id, track in TRACKS.items():
        if not track.get("tuned_model"):
            logger.error(f"Invalid configuration: Track {track_id} is missing 'tuned_model'")
            raise RuntimeError(f"Invalid configuration: Track {track_id} is missing 'tuned_model'")
        if not track.get("default_topic"):
            logger.error(f"Invalid configuration: Track {track_id} is missing 'default_topic'")
            raise RuntimeError(f"Invalid configuration: Track {track_id} is missing 'default_topic'")

# API Endpoints
@app.get("/")
async def root():
    """Return a welcome message for the root path."""
    return {
        "message": "Welcome to the Question Generation API",
        "documentation": "/docs",
        "endpoints": {
            "GET /tracks": "List available tracks",
            "GET /generate-questions": "Generate questions for single or multiple topics",
            "POST /generate-questions": "Generate questions with custom topics"
        }
    }

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    """Generate interview questions for single or multiple topics via POST request."""
    request_id = str(uuid.uuid4())
    with log_context(request_id):
        logger.info(f"Processing request: {request.dict()}")
        if request.difficulty not in ["beginner", "intermediate", "advanced"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty.")
        if not isinstance(request.num_questions, int) or request.num_questions < 1 or request.num_questions > 20:
            raise HTTPException(status_code=400, detail="Number of questions must be an integer between 1 and 20.")
        if not request.track_id and not request.topics:
            raise HTTPException(status_code=400, detail="Either track_id or topics must be provided.")

        generator = QuestionGenerator()
        try:
            generator.setup_environment()
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {str(e)}")

        # Determine selected topics
        selected_topics = request.topics or []
        if request.track_id:
            if request.track_id not in TRACKS:
                raise HTTPException(status_code=400, detail=f"Invalid track_id. Choose from {', '.join(TRACKS.keys())}.")
            # Use custom topics if provided; otherwise, fall back to default topic
            if not selected_topics:
                selected_topics = [TRACKS[request.track_id]["default_topic"]]

        if not selected_topics:
            raise HTTPException(status_code=400, detail="At least one topic must be provided when track_id is not provided or invalid.")

        topic_questions_list = []
        num_topics = len(selected_topics)
        questions_per_topic = request.num_questions // num_topics
        extra_questions = request.num_questions % num_topics

        for i, topic in enumerate(selected_topics):
            topic_questions_count = questions_per_topic + (1 if i < extra_questions else 0)
            if topic_questions_count == 0:
                logger.warning(f"Skipping topic '{topic}' as it has 0 questions allocated.")
                continue
            topic_questions = await generate_questions_for_topic(
                generator, topic, request.track_id, request.difficulty, topic_questions_count
            )
            topic_questions_list.append(topic_questions)

        return QuestionResponse(topics=topic_questions_list)

@app.get("/generate-questions", response_model=QuestionResponse)
async def generate_questions_get(
    track_id: Optional[str] = Query(None, description="Track ID (e.g., '1' for Flutter, '2' for ML, '3' for Backend.NET, '4' for Frontend)"),
    topics: Optional[str] = Query(None, description="Comma-separated list of topics (e.g., 'pandas,neural network,asp.net,react')"),
    difficulty: str = Query("beginner", description="Difficulty: beginner, intermediate, advanced"),
    num_questions: int = Query(3, ge=1, le=20, description="Number of questions (1 to 20)")
):
    """Generate interview questions for single or multiple topics via GET request."""
    request_id = str(uuid.uuid4())
    with log_context(request_id):
        logger.info(f"Processing GET request: track_id={track_id}, topics={topics}, difficulty={difficulty}, num_questions={num_questions}")
        if difficulty not in ["beginner", "intermediate", "advanced"]:
            raise HTTPException(status_code=400, detail="Invalid difficulty.")
        if not track_id and not topics:
            raise HTTPException(status_code=400, detail="Either track_id or topics must be provided.")

        # Determine selected topics
        selected_topics = [topic.strip() for topic in topics.split(",") if topic.strip()] if topics else []
        if track_id:
            if track_id not in TRACKS:
                raise HTTPException(status_code=400, detail=f"Invalid track_id. Choose from {', '.join(TRACKS.keys())}.")
            # Use custom topics if provided; otherwise, fall back to default topic
            if not selected_topics:
                selected_topics = [TRACKS[track_id]["default_topic"]]

        if not selected_topics:
            raise HTTPException(status_code=400, detail="At least one topic must be provided.")

        generator = QuestionGenerator()
        try:
            generator.setup_environment()
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {str(e)}")

        topic_questions_list = []
        num_topics = len(selected_topics)
        questions_per_topic = num_questions // num_topics
        extra_questions = num_questions % num_topics

        for i, topic in enumerate(selected_topics):
            topic_questions_count = questions_per_topic + (1 if i < extra_questions else 0)
            if topic_questions_count == 0:
                logger.warning(f"Skipping topic '{topic}' as it has 0 questions allocated.")
                continue
            topic_questions = await generate_questions_for_topic(
                generator, topic, track_id, difficulty, topic_questions_count
            )
            topic_questions_list.append(topic_questions)

        return QuestionResponse(topics=topic_questions_list)

@app.get("/tracks")
async def get_tracks():
    """Return available tracks."""
    return TRACKS

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
