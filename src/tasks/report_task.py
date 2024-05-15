import os
from llama_index.core.schema import MetadataMode 
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
from datetime import datetime
import sendgrid
import time
from sendgrid.helpers.mail import Mail, Email, Content, To
import markdown
load_dotenv(override=True)

prompt_template = """
You are a professional researcher in the field of AI. You are given a list of paper abstract in one day.
Your job is to summarize the trends and note out some most interesting papers in the list. 
Give the link to the full paper in the report.
===============
{daily_paper_content}
"""

recipient_emails = ["nlmbao2015@gmail.com", "ngohoangbach2004@gmail.com", "ngohoangbach4008@gmail.com"]

def send_daily_report_email(recipients, report_content):
    """Sends the generated daily report to a list of email recipients."""

    message = Mail(
        from_email=Email("nlmbao2015@gmail.com"),  # Replace with your verified SendGrid sender email
        to_emails=[To(email) for email in recipients],
        subject=f"Daily AI Research Report - {datetime.today().strftime('%Y-%m-%d')}",
        html_content=Content("text/html", report_content)  # Send as Markdown
    )

    try:
        sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(f"Email sent with status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending email: {e}")

def generate_daily_report(paper_list):
    today = datetime.today()  # Get the current date and time

    # Format the date as a string
    date = today.strftime("%Y-%m-%d")  # YYYY-MM-DD format

    daily_paper_content = "\n===============\n".join([paper.get_content(MetadataMode.LLM) for paper in paper_list])
    gemini_llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=os.environ["GOOGLE_API_KEY"])
    
    prompt = prompt_template.format(daily_paper_content=daily_paper_content)
    try:
        response = gemini_llm.complete(prompt)
        
    except Exception as e:
        time.sleep(120)
        response = gemini_llm.complete(prompt)
        
        
    with open(f"./outputs/daily_report_{date}.md", "w") as f:
        f.write(response.text)
    
    print("Generate Daily Report successfully!")
    send_daily_report_email(recipient_emails, response.text)
    print("Send email successfully!")
    
    return response.text
        
        
if __name__ == "__main__":
    send_daily_report_email(recipient_emails, markdown.markdown("Test email content"))

