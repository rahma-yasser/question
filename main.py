import logging
import json
import re
from typing import List, Optional
from datetime import datetime
import google.generativeai as genai
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import FastAPI, HTTPException, Query
import asyncio
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Question Generation API",
    description="API for generating interview questions and answers using Gemini models. Supports POST for detailed requests and GET for single or multi-topic queries.",
    version="1.0.4"
)

TRACKS = {
    "1": {
        "name": "flutter developer",
        "default_topic": "flutter developer"
    },
    "2": {
        "name": "machine learning",
        "default_topic": "machine learning"
    },
    "3": {
        "name": "backend dotnet",
        "default_topic": ".NET backend"
    },
    "4": {
        "name": "frontend",
        "default_topic": "frontend development"
    }
}

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
    track_id: Optional[str] = Field(None, description="Track ID (e.g., '1' for Flutter, '2' for Machine Learning, '3' for .NET Backend, '4' for Frontend)")
    topics: Optional[List[str]] = Field(None, description="List of custom topics (e.g., ['pandas', 'numpy', 'ASP.NET', 'React'])")
    difficulty: str = Field("beginner", description="Difficulty level: beginner, intermediate, or advanced")
    num_questions: int = Field(3, ge=1, le=20, description="Number of questions (1 to 20)")

class QuestionGenerator:
    """A system to generate questions and answers using Google Gemini API."""
    
    def __init__(self):
        """Initialize the question generator."""
        self.google_api_key = None
        self.question_model = None
        self.rate_limit_delay = 2  

    def setup_environment(self) -> None:
        """Load environment variables and configure Google API."""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            logger.error("GOOGLE_API_KEY environment variable is not set")
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=self.google_api_key)
        logger.info("Environment setup completed.")

    def initialize_models(self) -> None:
        """Initialize the question model."""
        try:
            self.question_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def generate_content(self, prompt: str, response_type: str = "text/plain") -> str:
        """Generate content using the Gemini API with retry on rate limits."""
        try:
            async with asyncio.timeout(10):  
                response = self.question_model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": response_type,
                        "temperature": 0.8 
                    }
                )
                return response.text
        except asyncio.TimeoutError:
            logger.error("Gemini API call timed out")
            raise HTTPException(status_code=504, detail="Gemini API request timed out")
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise

async def generate_questions_for_topic(
    generator: QuestionGenerator,
    topic: str,
    track_id: str = None,
    difficulty: str = "beginner",
    num_questions: int = 3
) -> TopicQuestions:
    """Generate questions and answers for a single topic using Gemini."""
    selected_topic = topic.lower().strip()
    if not selected_topic:
        logger.error("Topic cannot be empty.")
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    question_list = []

    try:
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
        if len(questions_data) < num_questions:
            logger.warning(f"Expected {num_questions} questions, got {len(questions_data)}")

        for q in questions_data:
            if not q.get("question") or not q.get("gemini_answer") or not isinstance(q.get("question"), str) or len(q["question"].strip()) == 0:
                logger.warning(f"Skipping invalid question: {q.get('question', 'None')}")
                continue

            question_list.append(Question(
                question=q.get("question", ""),
                gemini_answer=re.sub(r'[^\x00-\x7F]+', '', re.sub(r'\s+', ' ', q.get("gemini_answer", "").strip())),
                user_answer="",
                topic=selected_topic,
                classification=difficulty.capitalize()
            ))

        return TopicQuestions(topic=selected_topic, questions=question_list)

    except Exception as e:
        logger.error(f"Error generating questions for topic '{selected_topic}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.get("/")
async def root():
    """Return a welcome message for the root path."""
    return {
        "message": "Welcome to the Question Generation API",
        "documentation": "/docs",
        "endpoints": {
            "GET /tracks": "List available tracks",
            "GET /generate-questions": "Generate questions for a single topic",
            "GET /generate-multi-questions": "Generate questions for multiple topics",
            "POST /generate-questions": "Generate questions with custom topics"
        }
    }

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    """Generate interview questions based on provided parameters."""
    if request.difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty.")
    if not isinstance(request.num_questions, int) or request.num_questions < 1 or request.num_questions > 20:
        raise HTTPException(status_code=400, detail="Number of questions must be an integer between 1 and 20.")
    if not request.track_id and not request.topics:
        raise HTTPException(status_code=400, detail="Either track_id or topics must be provided.")

    generator = QuestionGenerator()
    try:
        generator.setup_environment()
        generator.initialize_models()
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {str(e)}")

    topic_questions_list = []
    if request.track_id:
        if request.track_id not in TRACKS:
            raise HTTPException(status_code=400, detail=f"Invalid track_id. Choose from {', '.join(TRACKS.keys())}.")
        selected_topics = request.topics if request.topics else [TRACKS[request.track_id]["default_topic"]]

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

    else:
        if not request.topics:
            raise HTTPException(status_code=400, detail="Topics list cannot be empty when track_id is not provided.")
        num_topics = len(request.topics)
        questions_per_topic = request.num_questions // num_topics
        extra_questions = request.num_questions % num_topics

        for i, topic in enumerate(request.topics):
            topic_questions_count = questions_per_topic + (1 if i < extra_questions else 0)
            if topic_questions_count == 0:
                logger.warning(f"Skipping topic '{topic}' as it has 0 questions allocated.")
                continue
            topic_questions = await generate_questions_for_topic(
                generator, topic, None, request.difficulty, topic_questions_count
            )
            topic_questions_list.append(topic_questions)

    return QuestionResponse(topics=topic_questions_list)

@app.get("/generate-questions", response_model=QuestionResponse)
async def generate_questions_get(
    track_id: Optional[str] = Query(None, description="Track ID (e.g., '1' for Flutter, '2' for ML, '3' for .NET Backend, '4' for Frontend)"),
    topic: Optional[str] = Query(None, description="Single topic (e.g., 'flutter', 'pandas', 'ASP.NET', 'React')"),
    difficulty: str = Query("beginner", description="Difficulty: beginner, intermediate, advanced"),
    num_questions: int = Query(3, ge=1, le=20, description="Number of questions (1 to 20)")
):
    """Generate interview questions via GET request for a single topic."""
    if difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty.")
    if not track_id and not topic:
        raise HTTPException(status_code=400, detail="Either track_id or topic must be provided.")

    generator = QuestionGenerator()
    try:
        generator.setup_environment()
        generator.initialize_models()
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {str(e)}")

    selected_topic = topic if topic else TRACKS.get(track_id, {}).get("default_topic")
    if not selected_topic:
        raise HTTPException(status_code=400, detail="Invalid track_id or topic.")

    topic_questions = await generate_questions_for_topic(
        generator, selected_topic, track_id, difficulty, num_questions
    )
    return QuestionResponse(topics=[topic_questions])

@app.get("/generate-multi-questions", response_model=QuestionResponse)
async def generate_multi_questions(
    track_id: Optional[str] = Query(None, description="Track ID (e.g., '1' for Flutter, '2' for ML, '3' for .NET Backend, '4' for Frontend)"),
    topics: str = Query(..., description="Comma-separated list of topics (e.g., 'pandas,neural network,ASP.NET,React')"),
    difficulty: str = Query("beginner", description="Difficulty: beginner, intermediate, advanced"),
    num_questions: int = Query(3, ge=1, le=20, description="Number of questions (1 to 20)")
):
    """Generate interview questions via GET request for multiple topics."""
    if difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty.")
    if not track_id and not topics:
        raise HTTPException(status_code=400, detail="Either track_id or topics must be provided.")

    selected_topics = [topic.strip() for topic in topics.split(",") if topic.strip()]
    if not selected_topics:
        raise HTTPException(status_code=400, detail="At least one topic must be provided.")

    generator = QuestionGenerator()
    try:
        generator.setup_environment()
        generator.initialize_models()
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
