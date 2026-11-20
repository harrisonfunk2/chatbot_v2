
// Cloud Run FastAPI backend
const API_URL = "https://chatbot-v2-10011149197.us-west1.run.app";

const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const messages = document.getElementById("messages");
const sendButton = document.getElementById("send-button");
const status = document.getElementById("status");
const welcome = document.getElementById("welcome");


// Add a user or assistant message to the conversation.
function addMessage(text, role) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    if (role === "assistant") {
        const label = document.createElement("div");
        label.className = "message-label";
        label.textContent = "Assistant";

        const content = document.createElement("div");
        content.className = "message-content";
        content.textContent = text;

        message.appendChild(label);
        message.appendChild(content);
    } else {
        message.textContent = text;
    }

    messages.appendChild(message);
    message.scrollIntoView({
        behavior: "smooth",
        block: "end"
    });

    return message;
}


// Display citations underneath the assistant's answer.
function addCitations(message, citations) {
    if (!citations || Object.keys(citations).length === 0) {
        return;
    }

    const container = document.createElement("div");
    container.className = "citations";

    const title = document.createElement("div");
    title.className = "citations-title";
    title.textContent = "Sources";

    container.appendChild(title);

    Object.entries(citations).forEach(([number, source]) => {
        const citation = document.createElement("div");
        citation.className = "citation";

        const description = typeof source === "string"
            ? source
            : JSON.stringify(source);

        citation.textContent = `[${number}] ${description}`;
        container.appendChild(citation);
    });

    message.appendChild(container);
}


// Send a question to the Cloud Run backend.
async function sendQuestion(question) {
    sendButton.disabled = true;
    input.disabled = true;

    status.textContent = "Generating an answer...";

    // Hide welcome screen after the first question.
    welcome.hidden = true;

    addMessage(question, "user");

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            let errorMessage;

            if (response.status === 429) {
                errorMessage =
                    "The chatbot has reached its usage limit. Please try again later.";
            } else if (response.status === 503) {
                errorMessage =
                    "The chatbot is temporarily unavailable.";
            } else if (response.status === 403) {
                errorMessage =
                    "Access to the chatbot is currently restricted.";
            } else {
                errorMessage =
                    `Unable to generate an answer (HTTP ${response.status}).`;
            }

            throw new Error(errorMessage);
        }

        const data = await response.json();

        const assistantMessage = addMessage(
            data.answer,
            "assistant"
        );

        addCitations(assistantMessage, data.citations);

        status.textContent = "";

    } catch (error) {
        console.error("Chat request failed:", error);

        if (error instanceof TypeError) {
            status.textContent =
                "Unable to connect to the chatbot server.";
        } else {
            status.textContent = error.message;
        }

    } finally {
        sendButton.disabled = false;
        input.disabled = false;
        input.focus();
    }
}


// Handle chat form submissions.
form.addEventListener("submit", (event) => {
    event.preventDefault();

    const question = input.value.trim();

    if (!question || sendButton.disabled) {
        return;
    }

    input.value = "";
    sendQuestion(question);
});


// Suggested questions fill the input field.
document.querySelectorAll(".example").forEach((button) => {
    button.addEventListener("click", () => {
        input.value = button.dataset.question;
        input.focus();
    });
});
