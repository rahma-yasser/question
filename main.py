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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Question Generation API",
    description="API for generating interview questions and answers using Gemini models. Supports POST for detailed requests and GET for simple queries.",
    version="1.0.1"
)

# Define tracks
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
    }
}

# Pydantic models for API requests and responses
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
    track_id: Optional[str] = Field(None, description="Track ID (e.g., '1' for Flutter, '2' for Machine Learning)")
    topics: Optional[List[str]] = Field(None, description="List of custom topics (e.g., ['pandas', 'numpy'])")
    difficulty: str = Field("beginner", description="Difficulty level: beginner, intermediate, or advanced")
    num_questions: int = Field(3, ge=1, le=5, description="Number of questions (1 to 5)")

class QuestionGenerator:
    """A system to generate questions and answers using Google Gemini API."""
    
    def __init__(self):
        """Initialize the question generator."""
        self.google_api_key = None
        self.question_model = None
        self.rate_limit_delay = 2  # Delay in seconds for free-tier (2 req/min)

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
            async with asyncio.timeout(10):  # Timeout after 10s
                response = self.question_model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": response_type}
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

    # Generate questions and answers in one API call
    try:
        question_prompt = (
            f"Generate a JSON array of {num_questions} {difficulty}-level questions about {selected_topic}, "
            f"each with a 'question' field and a 'gemini_answer' field. "
            f'Example: [{{"question": "What is a key feature of {selected_topic}?", "gemini_answer": "Answer here"}}]'
        )
        response_text = await generator.generate_content(
            question_prompt,
            response_type="application/json"
        )

        # Parse response
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

        # Process questions
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

@app.post("/generate-questions", response_model=QuestionResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    """Generate interview questions based on provided parameters."""
    if request.difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty.")
    if not isinstance(request.num_questions, int) or request.num_questions < 1 or request.num_questions > 5:
        raise HTTPException(status_code=400, detail="Number of questions must be an integer between 1 and 5.")
    if not request.track_id and not request.topics:
        raise HTTPException(status_code=400, detail="Either track_id or topics must be provided.")

    # Initialize generator
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
        if request.track_id == "2" and request.topics:
            selected_topics = request.topics
        else:
            selected_topics = [TRACKS[request.track_id]["default_topic"]]

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
    track_id: Optional[str] = Query(None, description="Track ID (e.g., '1' for Flutter, '2' for ML)"),
    topic: Optional[str] = Query(None, description="Single topic (e.g., 'flutter', 'pandas')"),
    difficulty: str = Query("beginner", description="Difficulty: beginner, intermediate, advanced"),
    num_questions: int = Query(3, ge=1, le=5, description="Number of questions (1 to 5)")
):
    """Generate interview questions via GET request for simple queries."""
    if difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty.")
    if not track_id and not topic:
        raise HTTPException(status_code=400, detail="Either track_id or topic must be provided.")

    # Initialize generator
    generator = QuestionGenerator()
    try:
        generator.setup_environment()
        generator.initialize_models()
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {str(e)}")

    # Determine topic
    selected_topic = topic if topic else TRACKS.get(track_id, {}).get("default_topic")
    if not selected_topic:
        raise HTTPException(status_code=400, detail="Invalid track_id or topic.")

    # Generate questions
    topic_questions = await generate_questions_for_topic(
        generator, selected_topic, track_id, difficulty, num_questions
    )
    return QuestionResponse(topics=[topic_questions])

@app.get("/tracks")
async def get_tracks():
    """Return available tracks."""
    return TRACKS

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
