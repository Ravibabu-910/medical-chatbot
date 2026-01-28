document.addEventListener("DOMContentLoaded", () => {

  const chatBox = document.getElementById("chat-box");
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");


  function addMessage(text, type) {

    const msg = document.createElement("div");

    msg.classList.add("message", type);

    msg.innerText = text;

    chatBox.appendChild(msg);

    chatBox.scrollTop = chatBox.scrollHeight;
  }


  async function sendMessage() {

    const question = input.value.trim();

    if (!question) return;

    addMessage(question, "user");

    input.value = "";

    // Loading message
    const loading = document.createElement("div");
    loading.classList.add("message", "bot", "loading");
    loading.innerText = "AI is thinking...";

    chatBox.appendChild(loading);

    chatBox.scrollTop = chatBox.scrollHeight;


    try {

      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: question
        })
      });

      const data = await response.json();

      loading.remove();

      if (data.answer) {
        addMessage(data.answer, "bot");
      }
      else {
        addMessage("No response from server", "bot");
      }

    }
    catch (err) {

      loading.remove();

      addMessage("❌ Server not running or connection failed", "bot");

      console.error(err);
    }

  }


  sendBtn.addEventListener("click", sendMessage);


  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  });

});
