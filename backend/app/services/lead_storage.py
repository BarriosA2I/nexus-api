"""
LEAD STORAGE SERVICE
====================
Persists captured leads to Notion database.
Sends email notifications for immediate follow-up.

Integration points:
- Notion API (via httpx)
- Resend API (email notifications)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

import httpx

logger = logging.getLogger("nexus.lead_storage")

# Configuration
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_LEADS_DATABASE_ID = os.getenv("NOTION_LEADS_DATABASE_ID")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
LEAD_NOTIFICATION_EMAIL = os.getenv("LEAD_NOTIFICATION_EMAIL", "gary@barriosa2i.com")
EMAIL_FROM = os.getenv("EMAIL_FROM", "leads@barriosa2i.com")

# Notion API headers
def get_notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }


async def save_lead_to_notion(lead_data: Dict, session_id: str = "unknown") -> Optional[str]:
    """
    Save a captured lead to the Notion database.

    Args:
        lead_data: Dict with email, name, company, industry, pain_point, interest_level
        session_id: The chat session ID for reference

    Returns:
        The created Notion page ID, or None if failed
    """
    if not NOTION_API_KEY or not NOTION_LEADS_DATABASE_ID:
        logger.warning("Notion not configured - lead not saved to database")
        return None

    try:
        # Build Notion page properties
        properties = {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": lead_data.get("name", "Unknown")[:100]
                        }
                    }
                ]
            },
            "Email": {
                "email": lead_data.get("email", "")
            },
            "Company": {
                "rich_text": [
                    {
                        "text": {
                            "content": lead_data.get("company", "Not specified")[:200]
                        }
                    }
                ]
            },
            "Industry": {
                "select": {
                    "name": lead_data.get("industry", "unknown")
                }
            },
            "Pain Point": {
                "rich_text": [
                    {
                        "text": {
                            "content": lead_data.get("pain_point", "Not specified")[:500]
                        }
                    }
                ]
            },
            "Interest Level": {
                "select": {
                    "name": lead_data.get("interest_level", "medium")
                }
            },
            "Status": {
                "select": {
                    "name": "New"
                }
            },
            "Source": {
                "rich_text": [
                    {
                        "text": {
                            "content": "Nexus Chat"
                        }
                    }
                ]
            },
            "Session ID": {
                "rich_text": [
                    {
                        "text": {
                            "content": session_id[:100]
                        }
                    }
                ]
            },
            "Captured At": {
                "date": {
                    "start": datetime.utcnow().isoformat()
                }
            }
        }

        # Create Notion page
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.notion.com/v1/pages",
                headers=get_notion_headers(),
                json={
                    "parent": {"database_id": NOTION_LEADS_DATABASE_ID},
                    "properties": properties
                },
                timeout=10.0
            )

            if response.status_code == 200:
                page_data = response.json()
                page_id = page_data.get("id")
                logger.info(f"✅ Lead saved to Notion: {page_id}")
                return page_id
            else:
                logger.error(f"Notion API error: {response.status_code} - {response.text}")
                return None

    except Exception as e:
        logger.error(f"Failed to save lead to Notion: {e}")
        return None


async def send_lead_notification(lead_data: Dict, notion_page_id: Optional[str] = None) -> bool:
    """
    Send email notification for new lead capture.

    Args:
        lead_data: Dict with lead information
        notion_page_id: Optional Notion page ID for direct link

    Returns:
        True if email sent successfully, False otherwise
    """
    if not RESEND_API_KEY:
        logger.warning("Resend not configured - notification not sent")
        return False

    try:
        email = lead_data.get("email", "unknown")
        industry = lead_data.get("industry", "unknown")
        interest = lead_data.get("interest_level", "medium")
        company = lead_data.get("company", "Not specified")
        pain_point = lead_data.get("pain_point", "Not specified")
        name = lead_data.get("name", "Unknown")

        # Build Notion link if available
        notion_link = ""
        if notion_page_id:
            # Remove dashes from page ID for URL
            clean_id = notion_page_id.replace("-", "")
            notion_link = f"\n\n🔗 View in Notion: https://notion.so/{clean_id}"

        # Email subject with key info
        subject = f"🎯 New Lead: {email} ({industry.replace('_', ' ').title()} - {interest.title()} Interest)"

        # Email body
        html_body = f"""
        <h2>🎯 New Lead Captured by Nexus</h2>

        <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Name</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{name}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Email</td>
                <td style="padding: 8px; border: 1px solid #ddd;"><a href="mailto:{email}">{email}</a></td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Company</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{company}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Industry</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{industry.replace('_', ' ').title()}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Pain Point</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{pain_point}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Interest Level</td>
                <td style="padding: 8px; border: 1px solid #ddd; color: {'green' if interest == 'high' else 'orange' if interest == 'medium' else 'gray'};">
                    <strong>{interest.upper()}</strong>
                </td>
            </tr>
        </table>

        <p style="margin-top: 20px;">
            <strong>Recommended Action:</strong> Respond within 24 hours for best conversion.
        </p>

        {f'<p><a href="https://notion.so/{notion_page_id.replace("-", "") if notion_page_id else ""}" style="background: #00CED1; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View in Notion</a></p>' if notion_page_id else ''}

        <hr style="margin-top: 30px;">
        <p style="color: #888; font-size: 12px;">
            Captured by Nexus Brain v4.1 | Barrios A2I
        </p>
        """

        # Plain text fallback
        text_body = f"""
🎯 New Lead Captured by Nexus

Name: {name}
Email: {email}
Company: {company}
Industry: {industry.replace('_', ' ').title()}
Pain Point: {pain_point}
Interest Level: {interest.upper()}

Recommended Action: Respond within 24 hours for best conversion.
{notion_link}

---
Captured by Nexus Brain v4.1 | Barrios A2I
        """

        # Send via Resend
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": EMAIL_FROM,
                    "to": [LEAD_NOTIFICATION_EMAIL],
                    "subject": subject,
                    "html": html_body,
                    "text": text_body
                },
                timeout=10.0
            )

            if response.status_code == 200:
                logger.info(f"✅ Lead notification sent to {LEAD_NOTIFICATION_EMAIL}")
                return True
            else:
                logger.error(f"Resend API error: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        logger.error(f"Failed to send lead notification: {e}")
        return False


async def process_captured_lead(lead_data: Dict, session_id: str = "unknown") -> Dict:
    """
    Full lead processing pipeline:
    1. Save to Notion
    2. Send email notification

    Args:
        lead_data: Dict from Claude's capture_lead tool
        session_id: Chat session ID

    Returns:
        Dict with processing results
    """
    results = {
        "notion_saved": False,
        "notion_page_id": None,
        "email_sent": False,
        "email": lead_data.get("email", "unknown")
    }

    # Step 1: Save to Notion
    notion_page_id = await save_lead_to_notion(lead_data, session_id)
    if notion_page_id:
        results["notion_saved"] = True
        results["notion_page_id"] = notion_page_id

    # Step 2: Send email notification
    email_sent = await send_lead_notification(lead_data, notion_page_id)
    results["email_sent"] = email_sent

    logger.info(f"📊 Lead processing complete: {results}")
    return results
