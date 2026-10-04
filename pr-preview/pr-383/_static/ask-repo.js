(function () {
  const REPO_URL = "https://github.com/Pressio/rom-tools-and-workflows";

  function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text) {
      node.textContent = text;
    }
    return node;
  }

  function currentPageReference() {
    if (window.location.protocol === "file:") {
      return window.location.pathname;
    }
    return window.location.href;
  }

  function buildPrompt(question) {
    const pageTitle = document.title || "ROM Tools and Workflows docs";
    const pageRef = currentPageReference();
    const trimmedQuestion = question.trim() || "Summarize the current page and explain the main ideas.";
    return [
      "I am reading the ROM Tools and Workflows documentation.",
      "",
      "Repository:",
      REPO_URL,
      "",
      "Current page:",
      pageTitle,
      pageRef,
      "",
      "Question:",
      trimmedQuestion,
      "",
      "Please answer using the documentation context from this page first, and use the repository structure and nearby source or docs files for additional context when needed. If you are uncertain, say what additional repo files or docs pages you would want to inspect."
    ].join("\n");
  }

  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const fallback = make("textarea");
    fallback.value = text;
    fallback.setAttribute("readonly", "readonly");
    fallback.style.position = "absolute";
    fallback.style.left = "-9999px";
    document.body.appendChild(fallback);
    fallback.select();
    document.execCommand("copy");
    document.body.removeChild(fallback);
  }

  function openChatGPT(prompt) {
    const url = "https://chatgpt.com/?q=" + encodeURIComponent(prompt);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  document.addEventListener("DOMContentLoaded", function () {
    const headerTarget = document.querySelector(".navbar-header-items__end");
    if (!headerTarget) {
      return;
    }

    const widget = make("div", "navbar-item ask-repo-widget");
    const form = make("form", "ask-repo-form");
    form.setAttribute("role", "search");

    const input = make("input", "ask-repo-input");
    input.type = "text";
    input.placeholder = "Ask ChatGPT about this repo";
    input.setAttribute("aria-label", "Ask ChatGPT about this repository");

    const button = make("button", "ask-repo-button", "Ask");
    button.type = "submit";

    form.appendChild(input);
    form.appendChild(button);
    widget.appendChild(form);
    headerTarget.prepend(widget);

    const overlay = make("div", "ask-repo-overlay");
    const dialog = make("div", "ask-repo-dialog");
    const dialogHeader = make("div", "ask-repo-dialog-header");
    const title = make("div", "ask-repo-dialog-title", "Ask ChatGPT about this repo");
    const close = make("button", "ask-repo-close", "Close");
    close.type = "button";
    close.setAttribute("aria-label", "Close Ask ChatGPT panel");

    const dialogForm = make("form", "ask-repo-dialog-form");
    const dialogInput = make("input", "ask-repo-dialog-input");
    dialogInput.type = "text";
    dialogInput.placeholder = "Ask about workflows, APIs, theory, or files";
    dialogInput.setAttribute("aria-label", "Ask a repository question");
    const actions = make("div", "ask-repo-dialog-actions");
    const copyButton = make("button", "ask-repo-dialog-button", "Copy Prompt");
    copyButton.type = "button";
    const openButton = make("button", "ask-repo-dialog-button ask-repo-dialog-button-secondary", "Open ChatGPT");
    openButton.type = "button";
    const hint = make("p", "ask-repo-hint", "This is a static helper. It copies or opens a suggested prompt for the current docs page.");
    const preview = make("pre", "ask-repo-preview");

    dialogHeader.appendChild(title);
    dialogHeader.appendChild(close);
    dialogForm.appendChild(dialogInput);
    actions.appendChild(copyButton);
    actions.appendChild(openButton);
    dialog.appendChild(dialogHeader);
    dialog.appendChild(dialogForm);
    dialog.appendChild(actions);
    dialog.appendChild(hint);
    dialog.appendChild(preview);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    function syncPreview() {
      preview.textContent = buildPrompt(dialogInput.value);
    }

    function openDialog(question) {
      overlay.classList.add("is-open");
      document.body.classList.add("ask-repo-open");
      dialogInput.value = question || input.value || "";
      syncPreview();
      dialogInput.focus();
      dialogInput.select();
    }

    function closeDialog() {
      overlay.classList.remove("is-open");
      document.body.classList.remove("ask-repo-open");
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      openDialog(input.value);
    });

    dialogForm.addEventListener("submit", function (event) {
      event.preventDefault();
      openChatGPT(buildPrompt(dialogInput.value));
    });

    dialogInput.addEventListener("input", syncPreview);

    openButton.addEventListener("click", function () {
      openChatGPT(buildPrompt(dialogInput.value));
    });

    copyButton.addEventListener("click", async function () {
      const original = copyButton.textContent;
      try {
        await copyText(buildPrompt(dialogInput.value));
        copyButton.textContent = "Copied";
        window.setTimeout(function () {
          copyButton.textContent = original;
        }, 1200);
      } catch (error) {
        copyButton.textContent = "Copy Failed";
        window.setTimeout(function () {
          copyButton.textContent = original;
        }, 1500);
      }
    });

    close.addEventListener("click", closeDialog);
    overlay.addEventListener("click", function (event) {
      if (event.target === overlay) {
        closeDialog();
      }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && overlay.classList.contains("is-open")) {
        closeDialog();
      }
    });
  });
})();
