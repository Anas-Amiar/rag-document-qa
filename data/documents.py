from rag.models import Document

KNOWLEDGE_BASE: list[Document] = [
    Document(
        doc_id="doc_001",
        title="Acme Platform Pricing",
        source="pricing_page",
        text=(
            "Acme Platform offers three pricing tiers: Starter, Growth, and Enterprise. "
            "The Starter plan costs $29 per month and supports up to 3 users and 10,000 API calls per month. "
            "The Growth plan costs $99 per month and supports up to 20 users and 100,000 API calls per month. "
            "The Enterprise plan has custom pricing negotiated directly with our sales team and offers "
            "unlimited users and API calls, plus a dedicated account manager and SLA guarantees. "
            "All plans include a 14-day free trial with no credit card required. "
            "Annual billing gives a 20% discount on Starter and Growth plans. "
            "There are no setup fees on any plan. Downgrading from Growth to Starter is allowed at any time "
            "and takes effect at the next billing cycle."
        ),
    ),
    Document(
        doc_id="doc_002",
        title="Acme Platform API Authentication",
        source="api_docs",
        text=(
            "Acme Platform uses API key authentication for all requests. "
            "Each API key is tied to a specific project and carries the permissions of the project owner. "
            "To generate an API key, navigate to Settings > API Keys in your dashboard and click 'Create Key'. "
            "API keys must be passed in the Authorization header as a Bearer token: "
            "'Authorization: Bearer YOUR_API_KEY'. "
            "Keys can be scoped to read-only or read-write access. "
            "If an API key is compromised, revoke it immediately from the dashboard — revocation takes effect "
            "within 60 seconds across all regions. "
            "API keys do not expire by default, but you can set an optional expiry date when creating them. "
            "Rate limits apply per API key: Starter keys are limited to 100 requests per minute, "
            "Growth keys to 500 requests per minute, and Enterprise keys have no hard rate limit."
        ),
    ),
    Document(
        doc_id="doc_003",
        title="Data Retention and Deletion Policy",
        source="legal_docs",
        text=(
            "Acme Platform retains user data for as long as an account is active. "
            "When an account is closed, all associated data is deleted within 30 days. "
            "You can request early deletion of your data by submitting a request through the Support portal. "
            "Backups are retained for 7 days after deletion to allow for accidental recovery. "
            "After 7 days, data is permanently and irreversibly purged from all backup systems. "
            "Data stored in the EU region is subject to GDPR retention rules. "
            "Acme does not sell or share user data with third parties except as required by law. "
            "Audit logs are retained for 90 days regardless of account status, as required for compliance. "
            "You can download a full export of your data at any time from Settings > Data Export."
        ),
    ),
    Document(
        doc_id="doc_004",
        title="Webhooks: Setup and Delivery",
        source="api_docs",
        text=(
            "Acme Platform supports webhooks for real-time event notifications. "
            "To register a webhook, go to Settings > Webhooks and provide a public HTTPS endpoint. "
            "Webhook payloads are signed using HMAC-SHA256 with your webhook secret. "
            "The signature is included in the X-Acme-Signature header of each delivery. "
            "Acme retries failed webhook deliveries up to 5 times with exponential backoff: "
            "first retry after 1 minute, then 5 minutes, 30 minutes, 2 hours, and 8 hours. "
            "If all retries fail, the event is marked as undelivered and logged in your Webhook Logs. "
            "Webhook endpoints must respond with HTTP 200 within 10 seconds; otherwise the delivery is "
            "counted as a failure. "
            "You can test webhook delivery from the dashboard using the 'Send Test Event' button."
        ),
    ),
    Document(
        doc_id="doc_005",
        title="Integrations: Slack and Jira",
        source="integrations_docs",
        text=(
            "Acme Platform integrates natively with Slack and Jira. "
            "The Slack integration sends notifications for key events: new alerts, status changes, and "
            "weekly summary reports. To connect Slack, go to Integrations > Slack and authorize the Acme app. "
            "You can configure which channels receive which event types from the Integrations settings page. "
            "The Jira integration allows you to create Jira issues directly from Acme alerts. "
            "Connect Jira by going to Integrations > Jira, entering your Jira domain and an API token "
            "generated from your Atlassian account settings. "
            "Once connected, any Acme alert can be pushed to Jira with a single click or via automation rules. "
            "Both integrations require admin-level access in Acme to configure but any team member can use "
            "them once set up."
        ),
    ),
    Document(
        doc_id="doc_006",
        title="Support and SLA",
        source="support_docs",
        text=(
            "Acme Platform provides support through three channels: in-app chat, email, and phone. "
            "Starter plan customers have access to email support with a 48-hour response time guarantee. "
            "Growth plan customers have access to email and in-app chat with a 24-hour response time. "
            "Enterprise customers receive 24/7 phone support with a 1-hour response time SLA and a "
            "dedicated account manager. "
            "All support requests are tracked in the Support portal at support.acme.io. "
            "For critical production incidents (P1), all customers can reach the emergency on-call team "
            "at emergency@acme.io regardless of their plan. "
            "Planned maintenance windows are announced at least 72 hours in advance via email and "
            "the Acme status page at status.acme.io. "
            "Acme guarantees 99.9% uptime for Growth and Enterprise plans, and 99.5% uptime for Starter."
        ),
    ),
]
