// IM Copilot Web UI

const API_BASE = '/api';

// DOM Elements
const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const chatMessages = document.getElementById('chat-messages');
const typingIndicator = document.getElementById('typing-indicator');
const newSessionBtn = document.getElementById('new-session-btn');
const sessionList = document.getElementById('session-list');
const currentThreadLabel = document.getElementById('current-thread');
const interruptModal = document.getElementById('interrupt-modal');
const modalTitle = document.getElementById('modal-title');
const modalBody = document.getElementById('modal-body');
const modalFooter = document.getElementById('modal-footer');

let currentThreadId = window.CURRENT_THREAD_ID || null;
let isProcessing = false;
let ws = null;
let statusPollTimer = null;

// Initialize
function init() {
    if (sessionList) {
        sessionList.addEventListener('click', handleSessionClick);
    }
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', createNewSession);
    }
    if (chatForm) {
        chatForm.addEventListener('submit', handleSubmit);
    }
    loadSessions();

    // Load history if we're on a specific thread page
    if (currentThreadId) {
        loadThreadHistory(currentThreadId);
        checkPendingInterrupt(currentThreadId);
    }

    // Connect WebSocket for real-time Feishu push
    if (window.CURRENT_USER && window.CURRENT_USER.open_id) {
        connectWebSocket(window.CURRENT_USER.open_id);
    }
}

// Check if there's a pending interrupt for this thread
async function checkPendingInterrupt(threadId) {
    try {
        const res = await fetch(`${API_BASE}/sessions/${threadId}/status`);
        const data = await res.json();
        if (data.status === 'interrupted') {
            handleInterrupt(data);
        } else if (data.status === 'processing') {
            setProcessing(true);
            startStatusPolling();
        } else if (data.status === 'complete') {
            handleComplete(data);
        } else if (data.status === 'error') {
            addMessage('assistant', data.message || '恢复执行时出错。');
        }
    } catch (err) {
        console.error('Failed to check interrupt status:', err);
    }
}

// Load sessions list
async function loadSessions() {
    try {
        const res = await fetch(`${API_BASE}/sessions`);
        const data = await res.json();
        renderSessions(data.sessions);
    } catch (err) {
        console.error('Failed to load sessions:', err);
    }
}

// Render session list
function renderSessions(sessions) {
    if (!sessionList) return;
    sessionList.innerHTML = sessions.map(s => {
        const sourceTag = s.source === 'feishu' ? '<span class="source-tag feishu">飞书</span>' : '';
        return `
        <div class="session-item ${s.thread_id === currentThreadId ? 'active' : ''}" data-thread-id="${s.thread_id}">
            <span class="session-name">${sourceTag}会话 ${s.thread_id.slice(0, 8)}</span>
            <button class="session-delete" data-thread-id="${s.thread_id}">×</button>
        </div>`;
    }).join('');
}

// Handle session click (select or delete)
function handleSessionClick(e) {
    const deleteBtn = e.target.closest('.session-delete');
    if (deleteBtn) {
        const threadId = deleteBtn.dataset.threadId;
        deleteSession(threadId);
        return;
    }

    const sessionItem = e.target.closest('.session-item');
    if (sessionItem) {
        const threadId = sessionItem.dataset.threadId;
        selectSession(threadId);
    }
}

// Select a session
function selectSession(threadId) {
    currentThreadId = threadId;
    window.CURRENT_THREAD_ID = threadId;
    window.location.href = `/chat/${threadId}`;
}

// Create new session
async function createNewSession() {
    try {
        const res = await fetch(`${API_BASE}/sessions`, { method: 'POST' });
        const data = await res.json();
        selectSession(data.thread_id);
    } catch (err) {
        console.error('Failed to create session:', err);
        alert('创建会话失败');
    }
}

// Delete session
async function deleteSession(threadId) {
    if (!confirm('确定要删除这个会话吗？')) return;
    try {
        await fetch(`${API_BASE}/sessions/${threadId}`, { method: 'DELETE' });
        if (threadId === currentThreadId) {
            window.location.href = '/';
        } else {
            loadSessions();
        }
    } catch (err) {
        console.error('Failed to delete session:', err);
    }
}

// Load thread history and render as chat messages
async function loadThreadHistory(threadId) {
    try {
        const res = await fetch(`${API_BASE}/sessions/${threadId}`);
        const data = await res.json();

        if (!data.history || data.history.length === 0) {
            return; // Keep welcome message
        }

        // Clear welcome message
        const welcome = chatMessages.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        const latestState = data.state || {};
        const completedApproval = (latestState.approvals || []).some(
            approval => approval.gate_name === 'plan_approval' && approval.status !== 'pending'
        );

        // Render each history step
        let lastRawMessage = null;
        const renderedThinking = new Set();
        const renderedInterrupts = new Set();
        const renderedArtifacts = new Set();
        let renderedSummary = false;
        for (const step of data.history) {
            const state = step.state || {};

            // User message: when raw_message changes
            if (state.raw_message && state.raw_message !== lastRawMessage) {
                addMessage('user', state.raw_message);
                lastRawMessage = state.raw_message;
            }

            // Thinking block for each node execution
            const thinkingKey = `${step.step}:${step.node}`;
            if (step.node && step.node !== 'input' && !renderedThinking.has(thinkingKey)) {
                addThinkingBlock(step.node, state, step.step);
                renderedThinking.add(thinkingKey);
            }

            // Interrupt block
            const interruptKey = step.interrupt ? `${step.step}:${step.interrupt.gate}` : null;
            const skipCompletedApproval = step.interrupt?.gate === 'plan_approval' && completedApproval;
            if (step.interrupt && !skipCompletedApproval && !renderedInterrupts.has(interruptKey)) {
                addInterruptBlock(step.interrupt);
                renderedInterrupts.add(interruptKey);
            }

            // Final summary
            if (state.summary && step.node === 'deliver' && !renderedSummary) {
                addMessage('assistant', state.summary);
                renderedSummary = true;
            }

            // Artifacts
            if (state.artifacts && Object.keys(state.artifacts).length > 0) {
                const newArtifacts = Object.fromEntries(
                    Object.entries(state.artifacts).filter(([key]) => !renderedArtifacts.has(key))
                );
                if (Object.keys(newArtifacts).length > 0) {
                    showArtifacts(newArtifacts);
                    Object.keys(newArtifacts).forEach(key => renderedArtifacts.add(key));
                }
            }
        }

        scrollToBottom();
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

// Add a thinking block showing node execution
function addThinkingBlock(nodeName, state, stepNum) {
    const nodeLabels = {
        intent: '意图识别',
        planner: '任务规划',
        clarification: '澄清问题',
        plan_approval: '计划审批',
        route_content: '路由分发',
        doc: '生成文档',
        whiteboard: '生成白板',
        slide: '生成PPT',
        content: '内容生成',
        verify: '质量验证',
        side_agent: '并行验证',
        route_after_verify: '结果路由',
        deliver: '总结交付',
    };

    const label = nodeLabels[nodeName] || nodeName;

    // Build thinking content based on node
    let details = '';
    if (nodeName === 'intent' && state.intent_type) {
        const intentLabels = {
            create_doc: '创建文档',
            create_whiteboard: '创建白板',
            create_slide: '创建PPT',
            create_multi: '多任务',
            chat: '聊天',
        };
        details = `识别意图: ${intentLabels[state.intent_type] || state.intent_type}`;
        if (state.intent_params && state.intent_params.topic) {
            details += ` | 主题: ${state.intent_params.topic}`;
        }
    } else if (nodeName === 'planner' && state.plan) {
        const planLabels = { doc: '文档', whiteboard: '白板', slide: 'PPT', deliver: '交付' };
        details = `执行计划: ${state.plan.map(s => planLabels[s] || s).join(' → ')}`;
    } else if ((nodeName === 'doc' || nodeName === 'whiteboard' || nodeName === 'slide' || nodeName === 'content') && state.artifacts) {
        const artifactKeys = Object.keys(state.artifacts);
        if (artifactKeys.length > 0) {
            details = `已生成: ${artifactKeys.map(k => state.artifacts[k].title || k).join(', ')}`;
        }
    } else if (nodeName === 'verify' && state.checks && state.checks.length > 0) {
        const latest = state.checks[state.checks.length - 1];
        const statusLabels = { pass: '通过', revise: '需修改', clarify: '需澄清' };
        details = `验证结果: ${statusLabels[latest.status] || latest.status}`;
        if (latest.reason) {
            details += ` | ${latest.reason}`;
        }
    } else if (nodeName === 'side_agent' && state.side_agent_results && state.side_agent_results.length > 0) {
        const latest = state.side_agent_results[state.side_agent_results.length - 1];
        const score = latest.validation_score;
        details = `并行验证: ${latest.task || 'unknown'}`;
        if (score !== undefined) {
            details += ` | 评分: ${score}`;
        }
        if (latest.status) {
            details += ` | 状态: ${latest.status}`;
        }
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'thinking-block';
    wrapper.innerHTML = `
        <div class="thinking-header">
            <span class="thinking-icon">⚙️</span>
            <span class="thinking-label">${label}</span>
            <span class="thinking-step">#${stepNum}</span>
        </div>
        ${details ? `<div class="thinking-details">${escapeHtml(details)}</div>` : ''}
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

// Add interrupt block to chat
function addInterruptBlock(interrupt) {
    const gateLabels = {
        clarification: '需要澄清',
        plan_approval: '计划审批',
    };

    let content = '';
    if (interrupt.questions && interrupt.questions.length > 0) {
        content = interrupt.questions.map((q, i) => `${i + 1}. ${q}`).join('\n');
    } else if (interrupt.plan && interrupt.plan.length > 0) {
        const planLabels = { doc: '文档', whiteboard: '白板', slide: 'PPT', deliver: '交付' };
        content = '计划: ' + interrupt.plan.map(s => planLabels[s] || s).join(' → ');
    }

    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant interrupt-message';
    wrapper.innerHTML = `
        <div class="message-avatar">⏸️</div>
        <div class="message-content">
            <p><strong>${gateLabels[interrupt.gate] || interrupt.gate}</strong></p>
            <pre>${escapeHtml(content)}</pre>
        </div>
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

// Handle form submit
async function handleSubmit(e) {
    e.preventDefault();
    if (isProcessing || !currentThreadId) return;

    const message = messageInput.value.trim();
    if (!message) return;

    // Add user message
    addMessage('user', message);
    messageInput.value = '';
    setProcessing(true);
    let keepProcessing = false;

    try {
        const res = await fetch(`${API_BASE}/sessions/${currentThreadId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ message }),
        });
        const data = await res.json();

        if (data.status === 'interrupted') {
            handleInterrupt(data);
        } else if (data.status === 'complete') {
            handleComplete(data);
        } else if (data.status === 'processing') {
            keepProcessing = true;
            startStatusPolling();
        }
    } catch (err) {
        console.error('Chat error:', err);
        addMessage('assistant', '抱歉，处理消息时出错了。请重试。');
    } finally {
        if (!keepProcessing) {
            setProcessing(false);
        }
    }
}

// Handle interrupt
function handleInterrupt(data) {
    const gate = data.gate;

    if (window.DEBUG_MODE && data.timing) {
        renderTiming(data.timing);
    }

    if (gate === 'plan_approval') {
        showPlanApprovalModal(data);
    } else if (gate === 'clarification') {
        showClarificationModal(data);
    } else {
        showGenericModal(data);
    }
}

// Show plan approval modal
function showPlanApprovalModal(data) {
    modalTitle.textContent = '计划审批';

    const planHtml = (data.plan || []).map(step => {
        const labels = { doc: '文档', whiteboard: '白板', slide: 'PPT', deliver: '交付' };
        return `<div class="plan-item">${labels[step] || step}</div>`;
    }).join('');

    modalBody.innerHTML = `
        <p>${data.message || '请审阅以下执行计划：'}</p>
        <div class="plan-list">${planHtml}</div>
        <p><strong>意图类型：</strong>${data.intent_type || '未知'}</p>
    `;

    modalFooter.innerHTML = `
        <button class="btn btn-secondary" onclick="rejectPlan()">拒绝</button>
        <button class="btn btn-success" onclick="approvePlan()">确认执行</button>
    `;

    interruptModal.style.display = 'flex';
}

// Show clarification modal
function showClarificationModal(data) {
    modalTitle.textContent = '需要澄清';

    const questions = data.questions || [];
    const inputsHtml = questions.map((q, i) => `
        <p><strong>问题 ${i + 1}：</strong>${q}</p>
        <input type="text" id="clarify-answer-${i}" placeholder="请输入回答...">
    `).join('');

    modalBody.innerHTML = `
        <p>${data.message || '为了更准确地制定计划，请回答以下问题：'}</p>
        ${inputsHtml}
    `;

    modalFooter.innerHTML = `
        <button class="btn btn-secondary" onclick="closeModal()">跳过</button>
        <button class="btn btn-success" onclick="submitClarification(${questions.length})">提交回答</button>
    `;

    interruptModal.style.display = 'flex';
}

// Show generic modal
function showGenericModal(data) {
    modalTitle.textContent = '需要确认';
    modalBody.innerHTML = `<p>${data.message || '请确认'}</p>`;
    modalFooter.innerHTML = `
        <button class="btn btn-success" onclick="submitGeneric(true)">确认</button>
    `;
    interruptModal.style.display = 'flex';
}

// Close modal
function closeModal() {
    interruptModal.style.display = 'none';
}

// Approve plan
async function approvePlan() {
    closeModal();
    setProcessing(true);
    let keepProcessing = false;
    try {
        const res = await fetch(`${API_BASE}/sessions/${currentThreadId}/resume`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ decision: JSON.stringify({ approved: true, feedback: '' }) }),
        });
        const data = await res.json();
        if (data.status === 'interrupted') {
            handleInterrupt(data);
        } else if (data.status === 'complete') {
            handleComplete(data);
        } else if (data.status === 'processing') {
            keepProcessing = true;
            startStatusPolling();
        }
    } catch (err) {
        console.error('Resume error:', err);
        addMessage('assistant', '恢复执行时出错。');
    } finally {
        if (!keepProcessing) {
            setProcessing(false);
        }
    }
}

// Reject plan
async function rejectPlan() {
    closeModal();
    setProcessing(true);
    let keepProcessing = false;
    try {
        const res = await fetch(`${API_BASE}/sessions/${currentThreadId}/resume`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ decision: JSON.stringify({ approved: false, feedback: '用户拒绝' }) }),
        });
        const data = await res.json();
        if (data.status === 'interrupted') {
            handleInterrupt(data);
        } else if (data.status === 'complete') {
            handleComplete(data);
        } else if (data.status === 'processing') {
            keepProcessing = true;
            startStatusPolling();
        }
    } catch (err) {
        console.error('Resume error:', err);
        addMessage('assistant', '恢复执行时出错。');
    } finally {
        if (!keepProcessing) {
            setProcessing(false);
        }
    }
}

// Submit clarification answers
async function submitClarification(count) {
    const answers = [];
    for (let i = 0; i < count; i++) {
        answers.push(document.getElementById(`clarify-answer-${i}`).value || '');
    }

    closeModal();
    setProcessing(true);
    let keepProcessing = false;
    try {
        const res = await fetch(`${API_BASE}/sessions/${currentThreadId}/resume`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ decision: JSON.stringify(answers) }),
        });
        const data = await res.json();
        if (data.status === 'interrupted') {
            handleInterrupt(data);
        } else if (data.status === 'complete') {
            handleComplete(data);
        } else if (data.status === 'processing') {
            keepProcessing = true;
            startStatusPolling();
        }
    } catch (err) {
        console.error('Resume error:', err);
        addMessage('assistant', '恢复执行时出错。');
    } finally {
        if (!keepProcessing) {
            setProcessing(false);
        }
    }
}

// Submit generic decision
async function submitGeneric(approved) {
    closeModal();
    setProcessing(true);
    let keepProcessing = false;
    try {
        const res = await fetch(`${API_BASE}/sessions/${currentThreadId}/resume`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ decision: JSON.stringify({ approved }) }),
        });
        const data = await res.json();
        if (data.status === 'interrupted') {
            handleInterrupt(data);
        } else if (data.status === 'complete') {
            handleComplete(data);
        } else if (data.status === 'processing') {
            keepProcessing = true;
            startStatusPolling();
        }
    } catch (err) {
        console.error('Resume error:', err);
        addMessage('assistant', '恢复执行时出错。');
    } finally {
        if (!keepProcessing) {
            setProcessing(false);
        }
    }
}

// Handle complete response
function handleComplete(data) {
    stopStatusPolling();
    const summary = data.summary || '处理完成';
    addMessage('assistant', summary);

    // Show artifacts if any
    const artifacts = data.artifacts || {};
    if (Object.keys(artifacts).length > 0) {
        showArtifacts(artifacts);
    }

    // Show timing panel in debug mode
    if (window.DEBUG_MODE && data.timing) {
        renderTiming(data.timing);
    }
}

function startStatusPolling() {
    if (statusPollTimer || !currentThreadId) return;
    statusPollTimer = setInterval(checkProcessingStatus, 3000);
}

function stopStatusPolling() {
    if (!statusPollTimer) return;
    clearInterval(statusPollTimer);
    statusPollTimer = null;
}

async function checkProcessingStatus() {
    if (!currentThreadId) return;
    try {
        const res = await fetch(`${API_BASE}/sessions/${currentThreadId}/status`);
        const data = await res.json();
        if (data.status === 'processing' || data.status === 'idle') return;
        stopStatusPolling();
        setProcessing(false);
        if (data.status === 'interrupted') {
            handleInterrupt(data);
        } else if (data.status === 'complete') {
            handleComplete(data);
        } else if (data.status === 'error') {
            addMessage('assistant', data.message || '恢复执行时出错。');
        }
    } catch (err) {
        console.error('Status polling error:', err);
    }
}

// Show artifacts
function showArtifacts(artifacts) {
    const artifactHtml = Object.entries(artifacts).map(([key, artifact]) => `
        <div class="artifact-card">
            <h4>${escapeHtml(artifact.title || key)} (${escapeHtml(artifact.kind || '')})</h4>
            <pre>${escapeHtml(artifact.preview || '')}</pre>
        </div>
    `).join('');

    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';
    wrapper.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <p><strong>生成的内容：</strong></p>
            <div class="artifacts">${artifactHtml}</div>
        </div>
    `;
    chatMessages.appendChild(wrapper);
    scrollToBottom();
}

// Add message to chat
function addMessage(role, text) {
    // Remove welcome message on first real message
    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.innerHTML = `
        <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
        <div class="message-content text-content">${escapeHtml(text)}</div>
    `;
    chatMessages.appendChild(msg);
    scrollToBottom();
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Scroll to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Set processing state
function setProcessing(processing) {
    isProcessing = processing;
    if (typingIndicator) {
        typingIndicator.style.display = processing ? 'flex' : 'none';
    }
    if (messageInput) {
        messageInput.disabled = processing;
    }
    const sendBtn = chatForm?.querySelector('.btn-send');
    if (sendBtn) {
        sendBtn.disabled = processing;
    }
}

// Expose functions to window for inline onclick handlers
window.closeModal = closeModal;
window.approvePlan = approvePlan;
window.rejectPlan = rejectPlan;
window.submitClarification = submitClarification;
window.submitGeneric = submitGeneric;

// WebSocket connection for real-time Feishu message push
function connectWebSocket(openId) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/${openId}`;
    ws = new WebSocket(url);

    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleWsMessage(data);
        } catch (e) {
            console.error('WS parse error:', e);
        }
    };

    ws.onclose = function() {
        setTimeout(() => connectWebSocket(openId), 3000);
    };

    ws.onerror = function() {
        ws.close();
    };
}

function handleWsMessage(data) {
    const threadId = data.thread_id;
    if (!threadId) return;

    if (data.type === 'message' && data.data) {
        if (threadId === currentThreadId) {
            addMessage(data.data.role || 'user', data.data.content || '');
        }
        loadSessions();
    } else if (data.type === 'complete' && data.data) {
        if (threadId === currentThreadId) {
            if (data.data.summary) {
                addMessage('assistant', data.data.summary);
            }
            if (data.data.artifacts && Object.keys(data.data.artifacts).length > 0) {
                showArtifacts(data.data.artifacts);
            }
        }
        loadSessions();
    } else if (data.type === 'node_update') {
        if (threadId === currentThreadId && data.data) {
            addThinkingBlock(data.data.node || 'unknown', {}, data.data.step || 0);
        }
    }
}

// Render timing waterfall chart (debug mode)
function renderTiming(timing) {
    const panel = document.getElementById('timing-panel');
    const header = document.getElementById('timing-header');
    const bars = document.getElementById('timing-bars');
    if (!panel || !header || !bars) return;

    const totalMs = timing.graph_duration_ms || 1;
    header.textContent = `全链路耗时: ${totalMs.toFixed(0)}ms`;

    const nodeLabels = {
        intent: '意图识别',
        planner: '任务规划',
        clarification: '澄清问题',
        plan_approval: '计划审批',
        route_content: '路由分发',
        doc: '生成文档',
        whiteboard: '生成白板',
        slide: '生成PPT',
        verify: '质量验证',
        side_agent: '并行验证',
        route_after_verify: '结果路由',
        deliver: '总结交付',
    };

    const statusColors = {
        success: '#4caf50',
        error: '#f44336',
        interrupted: '#ff9800',
        rate_limited: '#ff5722',
    };

    bars.innerHTML = (timing.nodes || []).map(n => {
        const pct = Math.max((n.duration_ms / totalMs) * 100, 2);
        const left = (n.start_ms / totalMs) * 100;
        const color = statusColors[n.status] || '#90a4ae';
        const label = nodeLabels[n.node] || n.node;
        return `<div class="timing-row">
            <span class="timing-node-name">${escapeHtml(label)}</span>
            <div class="timing-track">
                <div class="timing-bar" style="width:${pct}%;left:${left}%;background:${color}">
                    ${n.duration_ms.toFixed(0)}ms
                </div>
            </div>
        </div>`;
    }).join('');

    panel.style.display = 'block';
}

// Start
document.addEventListener('DOMContentLoaded', init);
