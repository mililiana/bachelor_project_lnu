(function () {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/@mdi/font@7.2.96/css/materialdesignicons.min.css';
    document.head.appendChild(link);

    const markedScript = document.createElement('script');
    markedScript.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    document.head.appendChild(markedScript);

    const katexCSS = document.createElement('link');
    katexCSS.rel = 'stylesheet';
    katexCSS.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css';
    document.head.appendChild(katexCSS);

    const katexScript = document.createElement('script');
    katexScript.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js';
    katexScript.async = false;
    document.head.appendChild(katexScript);

    const katexAutoRender = document.createElement('script');
    katexAutoRender.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js';
    katexAutoRender.async = false;
    document.head.appendChild(katexAutoRender);

    const widgetCSS = document.createElement('link');
    widgetCSS.rel = 'stylesheet';
    widgetCSS.href = 'widget.css';
    document.head.appendChild(widgetCSS);

    const container = document.createElement('div');
    container.className = 'lpnu-assistant-container';
    container.innerHTML = `
        <div class="assistant-chat-window" id="chatWindow" style="display: none;">
            <div class="chat-header">
                <span class="chat-title">LPNU ASSISTANT</span>
                <span class="mdi mdi-close close-chat" id="closeChat" style="cursor: pointer;"></span>
            </div>
            <div class="chat-messages" id="chatMessages">
            </div>
        </div>

        <div class="assistant-label">LPNU ASSISTANT</div>
        <div class="bot-icon-container" id="botIcon"></div>

        <div class="ask-me-pill">
            <textarea class="pill-input" id="pillInput" placeholder="ask me...." rows="1"></textarea>
            <span id="sendBtn" class="mdi mdi-arrow-right-circle-outline send-icon"></span>
        </div>
    `;
    document.body.appendChild(container);

    const input = document.getElementById('pillInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatWindow = document.getElementById('chatWindow');
    const chatMessages = document.getElementById('chatMessages');
    const closeBtn = document.getElementById('closeChat');

    input.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    input.addEventListener('focus', () => {
        container.classList.add('active');
        if (chatMessages.children.length > 0) {
            chatWindow.style.display = 'flex';
        }
    });

    document.addEventListener('click', (event) => {
        const isClickInside = container.contains(event.target);
        if (!isClickInside) {
            container.classList.remove('active');
            chatWindow.style.display = 'none';
        }
    });

    closeBtn.onclick = (e) => {
        if (e) e.stopPropagation();
        chatWindow.style.display = 'none';
        container.classList.remove('active');
        input.value = "";
        input.style.height = 'auto';
    };

    const appendMessage = (text, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-bubble ${sender}-bubble`;
        msgDiv.innerHTML = text;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    };

    const handleSend = (e) => {
        if (e) e.stopPropagation();
        const query = input.value.trim();
        if (query !== "") {
            appendMessage(query, 'user');
            chatWindow.style.display = 'flex';

            input.value = "";
            input.style.height = 'auto';
            input.disabled = true;

            const typingIndicator = appendMessage("typing ...", 'bot');

            fetch('http://localhost:5001/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ "user_query": query })
            })
                .then(res => {
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    return res.json();
                })
                .then(data => {
                    console.log('[LPNU Bot] response:', data);
                    typingIndicator.remove();

                    let htmlContent = data.answer || JSON.stringify(data);
                    if (window.marked && data.answer) {
                        let rawText = data.answer;
                        const mathBlocks = [];

                        rawText = rawText.replace(/(\$\$[\s\S]*?\$\$|\$[^$\n]+\$)/g, (match) => {
                            mathBlocks.push(match);
                            return `XYZMATHBLOCKXYY${mathBlocks.length - 1}YYZ`;
                        });

                        htmlContent = marked.parse(rawText);

                        htmlContent = htmlContent.replace(/XYZMATHBLOCKXYY(\d+)YYZ/g, (match, index) => {
                            return mathBlocks[parseInt(index, 10)];
                        });
                    }

                    const adrRegex = /(вул\.\s+[А-ЯІЇЄа-яіїє\-\'\']+(?:\s+[А-ЯІЇЄа-яіїє\-\'\']+)*,?\s*\d+(?:[а-яА-ЯіїєІЇЄ])?(?:,\s*\d+(?:[а-яА-ЯіїєІЇЄ])?)*)/gi;
                    let foundAddresses = [];
                    htmlContent = htmlContent.replace(adrRegex, (match) => {
                        foundAddresses.push(match);
                        const query = encodeURIComponent("Львів, " + match);
                        return `<a href="https://www.google.com/maps/search/?api=1&query=${query}" target="_blank" style="color: #3498DB; font-weight: bold; text-decoration: none; display: inline-flex; align-items: center; gap: 3px; border-bottom: 2px dashed #3498DB;"><span class="mdi mdi-map-marker" style="font-size: 1.1em;"></span>${match}</a>`;
                    });

                    if (foundAddresses.length > 0) {
                        const embedQuery = encodeURIComponent("Львів, " + foundAddresses[0]);
                        htmlContent += `
                        <div style="margin-top: 12px; border-radius: 8px; overflow: hidden; border: 1px solid rgba(0,0,0,0.1); box-shadow: 0 4px 10px rgba(0,0,0,0.08);">
                            <iframe
                                width="100%"
                                height="180"
                                frameborder="0"
                                scrolling="no"
                                marginheight="0"
                                marginwidth="0"
                                src="https://maps.google.com/maps?q=${embedQuery}&t=&z=15&ie=UTF8&iwloc=&output=embed">
                            </iframe>
                        </div>`;
                    }

                    if (data.sources && data.sources.length > 0) {
                        const docRefRegex = /(?:(?:\(|\[)?\bдокумент[а-яіїє]*\s+(\d+)(?:\)|\])?|\[\s*(\d+)\s*\])/gi;
                        htmlContent = htmlContent.replace(docRefRegex, (match, g1, g2) => {
                            const indexStr = g1 || g2;
                            const index = parseInt(indexStr, 10) - 1;
                            if (index >= 0 && index < data.sources.length) {
                                const source = data.sources[index];
                                const url = source.url && source.url !== '#' ? source.url : '#';
                                return ` <a href="${url}" target="_blank" title="${source.title}" style="color: #e67e22; font-weight: 500; text-decoration: underline; border-radius: 4px;">[${source.title}]</a> `;
                            }
                            return match;
                        });

                        htmlContent += '<div class="sources-list" style="margin-top: 10px; font-size: 0.9em; border-top: 1px solid rgba(0,0,0,0.1); padding-top: 10px;">';
                        htmlContent += '<strong style="color: #323296;">Релевантні джерела:</strong><ol style="margin: 5px 0; padding-left: 20px;">';

                        const seenTitles = new Set();
                        const displayedSources = [];
                        for (const src of data.sources) {
                            if (!seenTitles.has(src.title) && src.title !== "Без назви") {
                                seenTitles.add(src.title);
                                displayedSources.push(src);
                                if (displayedSources.length >= 3) break;
                            }
                        }

                        displayedSources.forEach((source, index) => {
                            htmlContent += `<li><a href="${source.url}" target="_blank" style="color: #323296; text-decoration: underline;">${source.title || 'Джерело ' + (index + 1)}</a></li>`;
                        });
                        htmlContent += '</ol></div>';
                    }

                    if (window.renderMathInElement) {
                        const tempDiv = document.createElement('div');
                        tempDiv.innerHTML = htmlContent;
                        renderMathInElement(tempDiv, {
                            delimiters: [
                                { left: '$$', right: '$$', display: true },
                                { left: '$', right: '$', display: false },
                                { left: '\\(', right: '\\)', display: false },
                                { left: '\\[', right: '\\]', display: true }
                            ],
                            throwOnError: false
                        });
                        htmlContent = tempDiv.innerHTML;
                    }

                    const msgDiv = appendMessage('', 'bot');

                    let i = 0;
                    let isTag = false;
                    let text = htmlContent;

                    function typeWriter() {
                        if (i < text.length) {
                            while (i < text.length) {
                                const char = text.charAt(i);
                                if (char === '<') isTag = true;
                                if (char === '>') isTag = false;

                                i++;
                                if (!isTag) break;
                            }

                            msgDiv.innerHTML = text.substring(0, i);
                            chatMessages.scrollTop = chatMessages.scrollHeight;

                            setTimeout(typeWriter, Math.random() * 10 + 5);
                        } else {
                            msgDiv.innerHTML = text;
                            input.disabled = false;
                            input.focus();
                        }
                    }

                    typeWriter();
                })
                .catch(err => {
                    console.error('[LPNU Bot] error:', err);
                    typingIndicator.remove();
                    appendMessage(`Error: ${err.message}. Is your backend running on :5001?`, 'bot');
                    input.disabled = false;
                });
        }
    };

    sendBtn.addEventListener('click', handleSend);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend(e);
        }
    });
})();
