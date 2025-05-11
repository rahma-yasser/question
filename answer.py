import logging
import json
from typing import List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
import os
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Question Evaluation API",
    description="API for evaluating user answers with scoring and educational links using Gemini models.",
    version="1.0.0"
)

# Pydantic models for input validation
class QuestionInput(BaseModel):
    question: str
    gemini_answer: str
    user_answer: str
    topic: str
    classification: str

class TopicInput(BaseModel):
    topic: str
    questions: List[QuestionInput]

class InputData(BaseModel):
    topics: List[TopicInput]

class QuestionOutput(BaseModel):
    question: str
    gemini_answer: str
    user_answer: str
    topic: str
    classification: str
    links: List[str]
    score_explanation: str
    score: float

class TopicOutput(BaseModel):
    topic: str
    questions: List[QuestionOutput]

class OutputData(BaseModel):
    topics: List[TopicOutput]

# Scoring and link generation prompts
SCORING_PROMPT = """
You are an expert evaluator AI helping score user answers in an educational interview system. Your goal is to assess whether the user demonstrates a basic understanding of the core concept, even if their phrasing, examples, or terminology differ from the ideal answer. Focus on the idea and understanding, not grammar, style, or advanced explanations.

Question: {question}
Reference Answer: {reference_answer}
User Answer: {user_answer}

Evaluation Guide:
- Score 0.9–1.0: Excellent answer — captures the core idea well, with clear and accurate details.
- Score 0.5–0.8: Partial understanding — correct direction but lacks clarity or key details.
- Score below 0.5: Misunderstood, unrelated, or significantly incorrect answer.

Instructions:
- Award a score between 0 and 1 (rounded to 4 decimals) based on how well the user understood the core concept.
- Provide a short, clear, and constructive explanation of the score, mentioning what was done well and what could be improved.
- Return the response in JSON format:
{
  "score": float,
  "score_explanation": "string"
}
"""

LINK_GENERATION_PROMPT = """
You are an educational assistant tasked with finding relevant, high-quality educational resources for a given question and its correct answer. Based on the question and reference answer provided, return 1–3 URLs to authoritative, educational websites or documentation that directly address the topic of the question. Ensure the links are specific to the concepts discussed (e.g., strings in programming for questions about strings) and avoid unrelated or generic sources. Prioritize official documentation, educational platforms, or reputable tutorials.

Question: {question}
Reference Answer: {reference_answer}

Provide the response in JSON format:
{
  "links": [
    "https://example-link1.com",
    "https://example-link2.com",
    "https://example-link3.com"
  ]
}
"""

class Evaluator:
    """A system to evaluate answers and generate educational links using Google Gemini API."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.google_api_key = None
        self.model = None
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
        """Initialize the Gemini model."""
        try:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def generate_content(self, prompt: str, response_type: str = "text/plain") -> str:
        """Generate content using the Gemini API with retry on rate limits."""
        try:
            async with asyncio.timeout(10):
                response = self.model.generate_content(
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
            raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")

    async def get_links(self, question: str, reference_answer: str) -> List[str]:
        """Generate educational links using Gemini API."""
        try:
            prompt = LINK_GENERATION_PROMPT.format(question=question, reference_answer=reference_answer)
            response_text = await self.generate_content(prompt, response_type="application/json")
            links_data = json.loads(response_text)
            return links_data.get("links", [])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse links response as JSON: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Failed to generate links: {str(e)}")
            return []

    async def evaluate_answer(self, question: str, reference_answer: str, user_answer: str) -> Dict:
        """Evaluate user answer and generate score, explanation, and links."""
        try:
            prompt = SCORING_PROMPT.format(
                question=question,
                reference_answer=reference_answer,
                user_answer=user_answer
            )
            response_text = await self.generate_content(prompt, response_type="application/json")
            try:
                response_data = json.loads(response_text)
                score = float(response_data.get("score", 0.0))
                explanation = response_data.get("score_explanation", "No explanation provided.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse scoring response as JSON: {str(e)}")
                # Fallback to simple scoring logic
                core_keywords = reference_answer.lower().split()
                user_words = user_answer.lower().split()
                matching_keywords = len(set(core_keywords) & set(user_words))
                keyword_ratio = matching_keywords / len(core_keywords) if core_keywords else 0

                if keyword_ratio >= 0.8:
                    score = round(0.9 + (keyword_ratio * 0.1), 4)
                    explanation = (
                        f"The user answer captures the core idea of '{question}' well, aligning closely with the reference answer. "
                        f"To improve, consider adding more specific details like those in the reference answer."
                    )
                elif keyword_ratio >= 0.4:
                    score = round(0.5 + (keyword_ratio * 0.3), 4)
                    explanation = (
                        f"The user answer shows partial understanding of '{question}' but misses some key details present in the reference answer. "
                        f"Try incorporating more specific terms or examples to enhance clarity."
                    )
                else:
                    score = round(keyword_ratio * 0.5, 4)
                    explanation = (
                        f"The user answer does not fully address the core concept of '{question}'. "
                        f"Review the reference answer and focus on including key concepts for a stronger response."
                    )

            links = await self.get_links(question, reference_answer)
            return {
                "score": score,
                "score_explanation": explanation,
                "links": links
            }
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")

@app.post("/evaluate", response_model=OutputData)
async def evaluate_answers(data: InputData):
    """Evaluate user answers, assign scores, and provide educational links."""
    evaluator = Evaluator()
    try:
        evaluator.setup_environment()
        evaluator.initialize_models()
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize evaluator: {str(e)}")

    try:
        output_topics = []
        for topic in data.topics:
            output_questions = []
            for question in topic.questions:
                evaluation = await evaluator.evaluate_answer(
                    question.question,
                    question.gemini_answer,
                    question.user_answer
                )
                output_question = QuestionOutput(
                    question=question.question,
                    gemini_answer=question.gemini_answer,
                    user_answer=question.user_answer,
                    topic=question.topic,
                    classification=question.classification,
                    links=evaluation["links"],
                    score_explanation=evaluation["score_explanation"],
                    score=evaluation["score"]
                )
                output_questions.append(output_question)
            output_topic = TopicOutput(
                topic=topic.topic,
                questions=output_questions
            )
            output_topics.append(output_topic)
        logger.info("Successfully evaluated answers")
        return OutputData(topics=output_topics)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Return a welcome message for the root path."""
    return {
        "message": "FastAPI evaluation API with Gemini AI link generation is running",
        "documentation": "/docs",
        "endpoints": {
            "POST /evaluate": "Evaluate user answers with scoring and educational links"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
