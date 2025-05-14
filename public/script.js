document.addEventListener("DOMContentLoaded", () => {

    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const chatbotConversation = document.getElementById("chatbot-conversation-container");

    async function progressConversation() {
        const question = userInput.value.trim();
        if (!question) return;
        // Add user message to chat
        const newHumanSpeechBubble = document.createElement("div");
        newHumanSpeechBubble.classList.add("speech", "speech-human");
        newHumanSpeechBubble.textContent = question;
        chatbotConversation.appendChild(newHumanSpeechBubble);
        userInput.value = "";
        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const data = await response.json();
            // Add AI response to chat
            const newAiSpeechBubble = document.createElement("div");
            newAiSpeechBubble.classList.add("speech", "speech-ai");
            newAiSpeechBubble.textContent = data.answer;
            chatbotConversation.appendChild(newAiSpeechBubble);
            chatbotConversation.scrollTop = chatbotConversation.scrollHeight;
        } catch (error) {
            console.error("Error:", error);
        }
    }

    sendBtn.addEventListener("click", progressConversation);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") progressConversation();
    });

 });

