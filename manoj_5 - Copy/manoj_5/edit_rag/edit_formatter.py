import os
import json
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from rag_it1.retrieval.vectorstore import get_vectorstore
from edit_rag.slack_button import send_job_desc
import requests
from slack_bolt.adapter.socket_mode import SocketModeHandler
from threading import Thread
from edit_rag.slack_button import app as slack_app, SLACK_APP_TOKEN 
import re
Thread(target=lambda: SocketModeHandler(slack_app, SLACK_APP_TOKEN).start(), daemon=True).start()
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
from maya_agent.database import insert_draft
from intent_entity_extractor.extractor import delete_user_draft
# Step 2: load the .env file
load_dotenv(dotenv_path=env_path)
CHANNEL_ID= None
LINKEDIN_ACCESS_TOKEN = os.getenv("LINKEDIN_ACCESS_TOKEN")
PERSON_URN = os.getenv("PERSON_URN")
SLACK_BOT = os.getenv("SLACK_BOT_TOKEN")

def send_slack_message(text):#Add Channel_id as a paramater
    headers = {
        "Authorization": f"Bearer {SLACK_BOT}",
        "Content-Type": "application/json; charset=utf-8"
    }
    
    data = {
        "channel": CHANNEL_ID,
        "text": text,
    }
    print(CHANNEL_ID)
    response = requests.post("https://slack.com/api/chat.postMessage", json=data, headers=headers)#Add username and user_id
 
    return response.json()





def delete_user_data(user_id: str):
    """
    Delete all documents from the vectorstore associated with a given user_id.
    """
    try:
        vectorstore = get_vectorstore()
        collection = vectorstore._collection  # Low-level access to Chroma collection

        # Perform deletion based on metadata filter
        deleted = collection.delete(where={"user_id": user_id})

        print(f"✅ Successfully deleted data for user_id: {user_id}")
        return {"status": "success", "user_id": user_id, "deleted": deleted}
    except Exception as e:
        print(f"❌ Failed to delete data for user_id: {user_id} - {str(e)}")
        return {"status": "error", "user_id": user_id, "error": str(e)}

def load_job_store(file_path="job_store.json"):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_job_store(data, file_path="job_store.json"):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def store_job_description(job_json_path="job_descp.json"):
    if not os.path.exists(job_json_path):
        return {"error": f"{job_json_path} not found"}

    with open(job_json_path, "r", encoding="utf-8") as f:
        job_data = json.load(f)

    user_id = job_data.get("user_id")
    job_desc = job_data.get("job_desc", "").strip()

    if not user_id or not job_desc:
        return {"error": "Missing user_id or job_desc"}

    job_store = load_job_store()
    job_store[user_id] = job_desc
    save_job_store(job_store)
    return {"status": "stored", "user_id": user_id}


def alter_job_description( reply,job_desc):
   
    reply_text = reply.strip()
    if not reply_text:
        return {"error": "Empty reply"}

    llm = ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template("""You are a smart and precise job description rewriting assistant.

Your task is to update a job description based on the user's reply.

Instructions:
Update only the following fields: title (apply normalization rules below), skills, experience, location.
Do not add any new details not found in the user reply.
Do not change formatting, tone, or overall structure of the original job description.
Your output must begin directly with the updated job description.

Job Title Normalization Rules:

Skill-to-Job Title Mapping:
UI/UX, Figma, Adobe XD, User Research → Designer
React, Angular, JavaScript, CSS → Frontend Developer
Node.js, Express, Django, Flask → Backend Developer
React + Django or Node → Full Stack Developer
Python, Machine Learning, Data Science → Data Scientist
SQL, MongoDB, PostgreSQL → Database Developer
Unity, Unreal, Game Dev → Game Developer
Swift, iOS → iOS Developer
Kotlin, Android → Android Developer
Flutter, React Native → Mobile Developer
AWS, Docker, Kubernetes → DevOps Engineer
Security, Cybersecurity → Security Engineer
Selenium, QA, Testing → QA Engineer
Scrum, Agile, PM → Project Manager
Digital Marketing, SEO → Marketing Specialist
Sales, CRM → Sales Representative
Content Writing, Copywriting → Content Writer
Video Editing → Video Editor
Blockchain, Web3 → Blockchain Developer
AI, NLP, Deep Learning → AI Engineer

Experience-Level Prefix:
0 to 2 years (including "fresher" or "entry level") → Junior
3 to 5 years (or "mid level") → Mid-level
5 to 7 years (or "senior") → Senior
8 to 11 years → Lead
12 or more years (or if title contains "manager", "head", or "director") → Principal

Examples:
“2 years frontend React” → Junior Frontend Developer
“5+ years backend with Django” → Senior Backend Developer
“12 years AI research” → Principal AI Engineer
“Entry level iOS” → Junior iOS Developer

Skill Refinement:
Group related tools such as React, JavaScript, CSS
Remove duplicate mentions of the same technology
Do not add any tools not found in the original or user reply

Format:
When rewriting, preserve original formatting, structure, and tone of the job description.

User's Reply:
{reply}

Original Job Description:
{job_desc}

                                              
Updated Job Description:
""")

    result = llm.invoke(
        prompt.format(reply=reply_text, job_desc=job_desc)
    ).content
     
    return {"new_job_description": result}


# ========================
# MAIN EXECUTION
# ========================
def post_job_to_linkedin(user_id,user_name,result):
    print("🚀 [post_job_to_linkedin]")


    post_text = (
        f"🚀 New Job Opportunity!\n\n"
        # f"📌 Title: {job_title}\n"
        # f"🧠 Experience: {experience}\n"
        # f"📍 Location: {location}\n"
        # f"🛠 Skills: {skills}\n\n"
        f"{result}\n\n"
        "#Hiring #JobOpening #Careers"
    )

    headers = {
        "Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0"
    }

    payload = {
        "author": PERSON_URN,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": post_text},
                "shareMediaCategory": "NONE"
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    try:
        res = requests.post("https://api.linkedin.com/v2/ugcPosts", headers=headers, json=payload)

        if res.status_code == 201:
            post_id = res.headers.get("x-restli-id", "unknown")
            post_url = f"@{user_name} -> https://www.linkedin.com/feed/update/{post_id}"
           
            print("jobposting url:")
            print(post_url)
            print("####")
            print(CHANNEL_ID)
            # ✅ Send to Slack
            slack_message = f"✅ Job posted successfully! <@{user_id}>, here’s your LinkedIn link:\nhttps://www.linkedin.com/feed/update/{post_id}"
            print(slack_message)
            send_slack_message(slack_message)

        else:
            print( f"❌ Failed: {res.status_code} - {res.text}")

    except Exception as e:
        print( f"❌ Exception: {e}")

  
def run_job_rewrite_pipeline(user_id, reply,job_desc,user_name,channel_id):
    

    result = alter_job_description(reply,job_desc)
    global CHANNEL_ID
    CHANNEL_ID= channel_id
    print(CHANNEL_ID)
    result=result["new_job_description"]
    print("\n Altered job description processed")
    # import uuid
    # job_id = str(uuid.uuid4())[:8] 


   


    # Fix file path to access edit_mode.json from root directory
    edit_mode_path = os.path.join(os.path.dirname(__file__), '..', 'edit_mode.json')
    
    try:
        with open(edit_mode_path, 'r') as f:
            content = f.read().strip()
            if not content:
                edit_mode = {}
            else:
                edit_mode = json.loads(content)

    except FileNotFoundError:
        edit_mode = {}
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in edit_mode.json: {e}")
        edit_mode = {}
    

    job_id = edit_mode[user_id]['job_id']
    # job=edit_mode[user_id]["job_data"]



    # use this channelid if there is an error in edge case

    # CHANNEL_ID= edit_mode[user_id]["channel_id"]

    job_title=""

    # Reset user's edit mode
    if user_id in edit_mode:
        edit_mode[user_id]["status"] = False
        edit_mode[user_id]["message"] = "null"
        edit_mode[user_id]["job_id"]="null"
        edit_mode[user_id]["channel_id"]="null"
        edit_mode[user_id]["user_name"]="null"
        edit_mode[user_id]["job_title"]="null"
        # edit_mode[user_id]["job_data"]="null"
    
    # Write back to file
    with open(edit_mode_path, 'w') as f:
        json.dump(edit_mode, f, indent=2)
    action = send_job_desc(channel_id, result, job_id, user_name, user_id)
    print(action)
    print(f"📤 Sending updated job description to Slack | job_id: {job_id}")
    
    if action == "approve":
        print("✅ Approved by user. Proceeding...")
        if user_id:
            delete_user_data(user_id)
        delete_user_draft(job_id,user_id)
        post_job_to_linkedin(user_id,user_name,result)
        
    elif action == "reject":
        print("🧹 User rejected. Resetting memory and halting job.")
        if user_id:
            delete_user_data(user_id)
        delete_user_draft(job_id,user_id)
        print(f"User selected: {action}")

    elif action =="edit":
        print("User clicked edit, initiating edit workflow")
            
        match = re.search(r"\*\*Job Title:\*\*\s*(.+)", result)
        if match:
            job_title = match.group(1)
            print(job_title)
            # Fix the file path to access edit_mode.json from root directory
        # edit_mode_path = os.path.join(os.path.dirname(__file__), '..', 'edit_mode.json')
        # try:
        #     with open(edit_mode_path,'r') as f:
        #         content = f.read().strip()
        #         if not content:
        #             # File is empty, initialize with empty dict
        #             edit_mode = {}
        #         else:
        #             edit_mode = json.loads(content)
        # except FileNotFoundError:
        #     # File doesn't exist, create with empty dict
        #     edit_mode = {}
        # except json.JSONDecodeError as e:
        #     print(f"Warning: Invalid JSON in edit_mode.json: {e}")
        #     edit_mode = {}
        
        # Store both status and the original message to be edited
        edit_mode[user_id] = {
            "status": True,
            "message": result,
            "job_id": job_id,
            "channel_id":CHANNEL_ID,
            "user_name":user_name,
            "job_title":job_title
            # "job_data":job,  
                # need to add job_type here
            
        }
        print("before write---------",edit_mode)


        with open(edit_mode_path,'w') as f:
            json.dump(edit_mode,f,indent=2)


        print("after edit------------",edit_mode)



            # Send the message to user asking for feedback
        message = f"✏ <@{user_id}>, I'm ready to help you edit the job description!\n\n"
        message += f"**Current Job Details:**\n"
        message += f"• Description: {result}\n"
        send_slack_message(message)
        print("Edit clicked")


    elif action =="draft":
        print("User selected draft, sent to draft function")
        print("---------draft-------")
        print(job_id)
        print(user_id)
        print(user_name)
        print(channel_id)
        print(job_title)
        print(result)

        
        insert_draft(
            job_id=job_id,  # ← you already generated it before calling send_job_desc
            user_id=user_id,
            username=user_name,
            channel_id=CHANNEL_ID,
            job_title=job_title,
            description=result
        )
        if user_id:
            delete_user_data(user_id)
        # delete_user_draft(job_id,user_id)
        draft_confirmation = f"✅ <@{user_id}>, your job posting has been saved as a draft!\n\n" \
                               f"📋 Draft Details:\n" \
                               f"• Job Title: {job_title}\n" \
                               f"• Job ID: `{job_id}`\n\n" \
                               f"• Say \"show my posts\" to view all your drafts\n" \
                            #    f"• Say \"edit {job_id}\" to modify this draft\n" \
                            #    f"• Say \"delete {job_id}\" to remove this draft"
            
                        #    f"💡 **To manage your drafts:**\n" \
                        #    f"• Say \"show my posts\" to view all your drafts\n" \
                        #    f"• Say \"edit {job_id}\" to modify this draft\n" \
                        #    f"• Say \"delete {job_id}\" to remove this draft"
        
        send_slack_message(draft_confirmation)
        # # Send confirmation message to Slack
        # draft_confirmation = f"✅ <@{user_id}>, your job posting has been saved as a draft!\n\n" \
        #                     f"📋 **Draft Details:**\n" \
        #                     f"• Job Title: {job.get('job_title', 'N/A')}\n" \
        #                     f"• Company: {job.get('company', 'N/A')}\n" \
        #                     f"• Job ID: `{job_id}`\n\n" \
                        #    f"💡 **To manage your drafts:**\n" \
                        #    f"• Say \"show my posts\" to view all your drafts\n" \
                        #    f"• Say \"edit {job_id}\" to modify this draft\n" \
                        #    f"• Say \"delete {job_id}\" to remove this draft"
        
        # send_slack_message(draft_confirmation)
        # # Set error to stop workflow from proceeding to LinkedIn posting
        # state["error"] = f"User selected: {action}"
        # state["job_result"] = f"Draft saved successfully: {job_id}"
     


# # Run only if script is executed directly
if __name__ == "__main__":
    user_id='12122'
    user_name='manoj'
    reply='change year of experience to 3 years'
    job_desc="Hey <@U09359UUX8X>, here's your job description:\n\n**Senior Backend Developer**\nWe are looking for a skilled Backend Developer with 5 years of experience in Python.\n**Requirements:**\n- Python programming\n- Database management\n- API development\n\n**Job Type:** Full-time\n\nDoes this look okay?"
    output = run_job_rewrite_pipeline(user_id, reply,job_desc,user_name)
    if output:
        print("\n Final Output:\n")
        print(output)