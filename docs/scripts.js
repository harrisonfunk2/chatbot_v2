// Replace with your actual Cloud Run service URL.
// Do not include /chat at the end.
const API_URL = "https://YOUR-CLOUD-RUN-SERVICE-URL";

const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const messages = document.getElementById("messages");
const sendButton = document.getElementById("send-button");
const status = document.getElementById("status");


function addMessage(text, role) {
    const message = document.createElement("div");

    message.className = `message ${role}`;
    message.textContent = text;

    messages.appendChild(message);
    message.scrollIntoView({ behavior: "smooth", block: "end" });
}


function displayCitations(citations) {
    if (!citations || Object.keys(citations).length === 0) {
        return;
    }

    const lines = Object.entries(citations).map(
        ([number, source]) => {
            const description = typeof source === "string"
                ? source
                : JSON.stringify(source);

            return `[${number}] ${description}`;
        }
    );

    addMessage("Sources:\n" + lines.join("\n"), "assistant");
}


async function sendQuestion(question) {
    sendButton.disabled = true;
    input.disabled = true;
    status.textContent = "Generating an answer...";

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
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        addMessage(data.answer, "assistant");
        displayCitations(data.citations);

        status.textContent = "";
    } catch (error) {
        console.error(error);
        status.textContent = "Unable to reach the chatbot right now.";
    } finally {
        sendButton.disabled = false;
        input.disabled = false;
        input.focus();
    }
}


form.addEventListener("submit", (event) => {
    event.preventDefault();

    const question = input.value.trim();

    if (!question) {
        return;
    }

    input.value = "";
    sendQuestion(question);
});


document.querySelectorAll(".example").forEach((button) => {
    button.addEventListener("click", () => {
        input.value = button.dataset.question;
        input.focus();
    });
});
