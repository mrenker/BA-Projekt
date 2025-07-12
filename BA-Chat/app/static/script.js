// RAG Chatbot Web UI JavaScript

class RAGChatbotUI {
    constructor() {
        this.apiBase = window.location.origin;
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.chatForm = document.getElementById('chatForm');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.modelSelect = document.getElementById('modelSelect');
        
        this.initializeEventListeners();
        this.updateSystemStatus();
        this.fetchAndPopulateModels();
        // Auto-refresh status every 30 seconds
        setInterval(() => this.updateSystemStatus(), 30000);
    }

    initializeEventListeners() {
        // Chat form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // Clear chat button
        document.getElementById('clearChat').addEventListener('click', () => {
            this.clearChat();
        });

        // Refresh status button
        document.getElementById('refreshStatus').addEventListener('click', () => {
            this.updateSystemStatus();
        });

        // Auto-scroll chat on window resize
        window.addEventListener('resize', () => {
            this.scrollToBottom();
        });

        // Enter key handling for better UX
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }

    async fetchAndPopulateModels() {
        if (!this.modelSelect) return;
        try {
            const response = await fetch(`${this.apiBase}/models`);
            if (!response.ok) throw new Error('Failed to fetch models');
            const data = await response.json();
            const models = data.data || [];
            this.modelSelect.innerHTML = '';
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.id;
                this.modelSelect.appendChild(option);
            });
        } catch (e) {
            this.modelSelect.innerHTML = '<option value="">(Failed to load models)</option>';
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        const selectedModel = this.modelSelect ? this.modelSelect.value : undefined;
        // Add user message to chat
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.showTypingIndicator();
        this.setInputDisabled(true);

        try {
            const response = await fetch(`${this.apiBase}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message, model: selectedModel })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.hideTypingIndicator();
            
            // Add bot response to chat
            this.addMessage('bot', data.response, {
                sources: data.sources,
                verificationPassed: data.verification_passed,
                generationAttempts: data.generation_attempts
            });

        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('system', `Error: ${error.message}`, { isError: true });
            this.showToast('Error sending message', 'error');
        } finally {
            this.setInputDisabled(false);
            this.messageInput.focus();
        }
    }

    addMessage(type, content, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (type === 'user') {
            messageContent.innerHTML = `<i class="fas fa-user me-2"></i>${this.escapeHtml(content)}`;
        } else if (type === 'bot') {
            messageContent.innerHTML = `<i class="fas fa-robot me-2"></i>${this.escapeHtml(content)}`;
        } else if (type === 'system') {
            messageContent.innerHTML = `<i class="fas fa-info-circle me-2"></i>${this.escapeHtml(content)}`;
        }

        messageDiv.appendChild(messageContent);

        // Add metadata for bot messages
        if (type === 'bot' && (metadata.sources || metadata.verificationPassed !== undefined)) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';

            const timeSpan = document.createElement('span');
            timeSpan.className = 'message-time';
            timeSpan.textContent = new Date().toLocaleTimeString();

            const statusDiv = document.createElement('div');
            statusDiv.className = 'verification-status';

            if (metadata.verificationPassed !== undefined) {
                const verificationIcon = metadata.verificationPassed ? 
                    '<i class="fas fa-check-circle verification-passed"></i>' :
                    '<i class="fas fa-exclamation-triangle verification-failed"></i>';
                
                const attemptsText = metadata.generationAttempts > 1 ? 
                    ` (${metadata.generationAttempts} attempts)` : '';
                
                statusDiv.innerHTML = `${verificationIcon} Verified${attemptsText}`;
            }

            metaDiv.appendChild(timeSpan);
            metaDiv.appendChild(statusDiv);
            messageDiv.appendChild(metaDiv);

            // Add sources if available
            if (metadata.sources && metadata.sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'message-sources';
                sourcesDiv.innerHTML = '<strong>Sources:</strong><br>';
                
                metadata.sources.forEach(source => {
                    const sourceSpan = document.createElement('span');
                    sourceSpan.className = 'source-item';
                    sourceSpan.textContent = source;
                    sourcesDiv.appendChild(sourceSpan);
                });

                messageDiv.appendChild(sourcesDiv);
            }
        }

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showTypingIndicator() {
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.style.display = 'none';
    }

    setInputDisabled(disabled) {
        this.messageInput.disabled = disabled;
        const sendButton = document.getElementById('sendButton');
        
        if (disabled) {
            sendButton.classList.add('btn-loading');
            sendButton.innerHTML = '';
        } else {
            sendButton.classList.remove('btn-loading');
            sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }

    scrollToBottom() {
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="message system-message">
                <div class="message-content">
                    <i class="fas fa-robot me-2"></i>
                    Chat cleared! Ask me anything about the documents in the knowledge base.
                </div>
            </div>
        `;
    }

    async updateSystemStatus() {
        try {
            // Update health status
            const healthResponse = await fetch(`${this.apiBase}/health`);
            const healthData = await healthResponse.json();

            this.updateHealthIndicator(healthResponse.ok, healthData);

            if (healthResponse.ok) {
                // Update verification status
                const verificationResponse = await fetch(`${this.apiBase}/verification/status`);
                if (verificationResponse.ok) {
                    const verificationData = await verificationResponse.json();
                    this.updateVerificationIndicator(verificationData);
                }

                // Update document count
                this.updateDocumentCount();
            }

        } catch (error) {
            this.updateHealthIndicator(false, { error: error.message });
        }
    }

    updateHealthIndicator(isHealthy, data) {
        const statusIcon = document.querySelector('#systemStatus i');
        const statusText = document.getElementById('statusText');
        const healthIndicator = document.getElementById('healthIndicator');

        if (isHealthy) {
            statusIcon.className = 'fas fa-circle status-online';
            statusText.textContent = 'Online';
            healthIndicator.className = 'badge bg-success';
            healthIndicator.textContent = 'Healthy';
        } else {
            statusIcon.className = 'fas fa-circle status-offline';
            statusText.textContent = 'Offline';
            healthIndicator.className = 'badge bg-danger';
            healthIndicator.textContent = 'Error';
        }
    }

    updateVerificationIndicator(data) {
        const verificationIcon = document.querySelector('#verificationStatus i');
        const verificationText = document.getElementById('verificationText');
        const verificationIndicator = document.getElementById('verificationIndicator');

        if (data.verification_enabled) {
            verificationIcon.className = 'fas fa-shield-alt status-online';
            verificationText.textContent = 'Enabled';
            verificationIndicator.className = 'badge bg-success';
            verificationIndicator.textContent = 'Enabled';
        } else {
            verificationIcon.className = 'fas fa-shield-alt status-warning';
            verificationText.textContent = 'Disabled';
            verificationIndicator.className = 'badge bg-warning';
            verificationIndicator.textContent = 'Disabled';
        }
    }

    async updateDocumentCount() {
        try {
            const response = await fetch(`${this.apiBase}/collection/count`);
            if (response.ok) {
                const data = await response.json();
                document.getElementById('docCountNumber').textContent = data.count;
            }
        } catch (error) {
            console.error('Error updating document count:', error);
        }
    }

    showToast(message, type = 'info') {
        const toastContainer = document.querySelector('.toast-container');
        const toast = document.getElementById('toastTemplate').cloneNode(true);
        toast.id = 'toast-' + Date.now();

        // Set toast type
        const icon = toast.querySelector('i');
        const title = toast.querySelector('.me-auto');
        
        switch (type) {
            case 'success':
                icon.className = 'fas fa-check-circle text-success me-2';
                title.textContent = 'Success';
                break;
            case 'error':
                icon.className = 'fas fa-exclamation-circle text-danger me-2';
                title.textContent = 'Error';
                break;
            case 'warning':
                icon.className = 'fas fa-exclamation-triangle text-warning me-2';
                title.textContent = 'Warning';
                break;
            default:
                icon.className = 'fas fa-info-circle text-primary me-2';
                title.textContent = 'Info';
        }

        toast.querySelector('.toast-body').textContent = message;
        toastContainer.appendChild(toast);

        const bsToast = new bootstrap.Toast(toast, {
            autohide: true,
            delay: 5000
        });

        bsToast.show();

        // Remove toast element after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RAGChatbotUI();
}); 