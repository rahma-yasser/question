import json
import re
import os
import asyncio
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import typing_extensions as typing
# import google as genai  # Fallback import
from google import genai

import uvicorn
import logging

# Configure logging for Heroku debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

covered_soft_skills = [
    "Communication", "Teamwork", "Conflict Resolution",
    "Time Management", "Adaptability", "Leadership",
    "Problem Solving", "Emotional Intelligence"
]
KEY = "AIzaSyAoEnHcllqAerkW7yxfUlUYcAlZRgoOwOg"
# Configure Gemini API client
# try:
    # genai.configure(api_key=KEY)
    # global client# = genai.Client(api_key=KEY)
client = genai.Client(api_key=KEY)

    # logger.info("Google Gemini API configured successfully")
# except Exception as e:
    # logger.error(f"Failed to configure Gemini API: {e}")

class SkillBreakdown(BaseModel):
    clarity: int
    example_quality: int
    structure: int
    outcome: int

class EvaluationInfo(BaseModel):
    score: int
    skill_breakdown: SkillBreakdown
    strengths: typing.List[str]
    weaknesses: typing.List[str]
    feedback: str

def parse_response(text):
    """Robust JSON parsing with error handling"""
    try:
        cleaned = re.sub(r'[\x00-\x1F]|```json|```', '', text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {text[:200]}... Error: {e}")
        return {"error": "Evaluation failed"}

def evaluate_response(question, answer, target_skill):
    """Returns JSON-formatted string of user's evaluation"""
    prompte = f"""
    You are a soft skills interviewer evaluating a user's response.

    Your role is to assess how reasonably and relevantly the response reflects the skill: **{target_skill}**.

    Use the schema provided to guide your evaluation. Each requirement must be completed.

    - Keep your tone supportive, understanding, and concise.
    - Focus more on how well the response reflects an **attempt** to demonstrate the skill, rather than perfection.
    - If the response is unrelated or completely off-topic, give a lower evaluation.
    - Do not leave any field empty.
    - All numeric scores must be between 0 and 10, based on overall effort, clarity, and relevance.

    Question: {question}

    Response: {answer}
    """
    try:
        # response = genai.GenerativeModel("gemini-2.0-flash").generate_content(
        #     contents=[prompt],
        #     generation_config={"response_mime_type": "application/json", "response_schema": EvaluationInfo}
        # )
        response=client.models.generate_content(model="gemini-2.0-flash",contents=[prompte], config={"response_mime_type": "application/json","response_schema":EvaluationInfo })

        logger.info("Successfully generated evaluation response")
        return response.text
    except Exception as e:
        logger.error(f"Error in Gemini API call: {e}")
        return json.dumps({"error": f"Evaluation failed due to API error: {str(e)}"})

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "This is the home page. Use /soft/{num_q} for WebSocket interview."}

@app.get("/favicon.ico")
async def favicon():
    favicon_path = "favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")
#################################################################################################################################################
@app.websocket("/soft/{num_q}")
async def soft(num_q: int, websocket: WebSocket):
    await websocket.accept()
    evaluation = []
    questions = []
    answer = ""
    try:
        # interviewer = genai.GenerativeModel("gemini-2.0-flash")
        interviewer = client.chats.create(model="gemini-2.0-flash")

        logger.info("Initialized Gemini model for interview")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model: {e}")
        await websocket.send_json({"error": f"Failed to initialize interviewer: {str(e)}"})
        await websocket.close()
        return

    prompt = (
            "You are soft skills interviewer asking related questions to user answers"
            "make it like real life soft skills interview conversation"
            "The question should be based on the previous user answer. "
            f"Here are the soft skills you should ask about: {', '.join(covered_soft_skills)}. note you it is okay to ask from them randomly depending on the user previous answer"
            "Avoid asking about the same soft skill again. "
            "Return the result in JSON format as {question: str ,target_skill: str}."
            "Now you should start with just the question avoid adding any other text"
        )

    async def keep_alive():
        try:
            while True:
                await websocket.send_json({"type": "ping"})
                logger.info("Ping sent")
                await asyncio.sleep(20)
        except Exception as e:
            logger.info(f"Keep-alive stopped: {e}")

    ping_task = asyncio.create_task(keep_alive())

    try:
        for i in range(num_q):
            logger.info(f"Generating question {i+1}/{num_q}")
            if not questions:
                # response = interviewer.generate_content(prompt)
                response = interviewer.send_message(prompt)

            else:
                # response = interviewer.generate_content(questions[-1]["question"] + "\nUser Answer: " + answer)
                response = interviewer.send_message(answer)

            
            interviewer_question_parsed = parse_response(response.text)
            print(interviewer_question_parsed,"\n\n ")
            if "error" in interviewer_question_parsed:
                logger.error("Failed to generate question")
                await websocket.send_json({"error": "Failed to generate question","text":interviewer_question_parsed})
                break

            questions.append(interviewer_question_parsed)
            await websocket.send_json(interviewer_question_parsed)
            logger.info(f"Sent question: {interviewer_question_parsed['question']}")

            try:
                recev = await websocket.receive_json()
                answer = recev.get("text", "")
                logger.info(f"Received answer: {answer[:50]}...")
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error receiving WebSocket data: {e}")
                await websocket.send_json({"error": f"Failed to receive answer: {str(e)}"})
                break

            evaluation_result = evaluate_response(
                interviewer_question_parsed["question"],
                answer,
                interviewer_question_parsed["target_skill"]
            )
            parsed_evaluation = json.loads(evaluation_result)
            evaluation.append(parsed_evaluation)
            await websocket.send_json(parsed_evaluation)
            logger.info("Sent evaluation result")
        # print(evaluation)
        if evaluation:
            df = pd.json_normalize(evaluation)
            avgs = df.select_dtypes(include='number').mean().to_json(indent=4)
            # avgs=str({"avgs":avgs})
            # print(avgs,"#"*40)
            avgs_json = {"avgs": json.loads(avgs)}  # Ensure 'avgs' is parsed JSON, not a string
            print(json.dumps(avgs_json, indent=4), "#"*40)
            await websocket.send_json(avgs_json)

            # await websocket.send_json(json.loads(avgs))
            # await websocket.send_json(avgs)
            logger.info("Sent average scores")
        else:
            await websocket.send_json({"error": "No evaluations to average"})
            logger.warning("No evaluations to average")

    except Exception as e:
        logger.error(f"Error in WebSocket loop: {e}")
        await websocket.send_json({"error": f"Interview terminated due to error: {str(e)}"})
    finally:
        ping_task.cancel()
        try:
            await ping_task
        except asyncio.CancelledError:
            pass
        await websocket.close()
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
