// Theme handling
(function initTheme(){
    const saved = localStorage.getItem('theme') || 'light';
    if (saved === 'dark') document.documentElement.setAttribute('data-theme','dark');
    const buttons = document.querySelectorAll('#theme-toggle');
    const setIcon = (btn) => { if (!btn) return; btn.textContent = document.documentElement.getAttribute('data-theme') === 'dark' ? '☀️' : '🌙'; };
    buttons.forEach(setIcon);
    document.addEventListener('click', (e)=>{
        if (e.target && e.target.id === 'theme-toggle'){
            const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
            const next = cur === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next === 'dark' ? 'dark' : '');
            localStorage.setItem('theme', next);
            document.querySelectorAll('#theme-toggle').forEach(setIcon);
        }
    });
})();

// Track conversation log for saving later
const conversationLog = [];
let sessionStart = new Date().toISOString();
let lastParsed = null;
let allEntries = []; // Global store for unique entries
let lastRawText = '';

// Modal handling and raw JSON toggler (modal)
const toolsFab = document.getElementById('tools-fab');
const toolsModal = document.getElementById('tools-modal');
const toolsModalClose = document.getElementById('tools-modal-close');
const rawPre = document.getElementById('json-content');
const rawBtn = document.getElementById('toggle-raw-btn');
if(toolsFab) toolsFab.addEventListener('click', ()=>{ toolsModal.classList.add('show'); });
if(toolsModalClose) toolsModalClose.addEventListener('click', ()=>{ toolsModal.classList.remove('show'); });
if(toolsModal) toolsModal.addEventListener('click', (e)=>{ if (e.target === toolsModal) toolsModal.classList.remove('show'); });
if(rawBtn) rawBtn.addEventListener('click', ()=>{
    const hidden = rawPre.classList.toggle('hidden');
    rawBtn.textContent = hidden ? 'Show' : 'Hide';
});

// Transition from start screen to main app
const startScreen = document.getElementById('start-screen');
const mainApp = document.getElementById('main-app');
function transitionToMain(){
    if (mainApp.classList.contains('hidden')){
        startScreen.classList.add('fade-out');
        setTimeout(()=>{ startScreen.style.display = 'none'; mainApp.classList.remove('hidden'); }, 200);
    }
}
// Start screen controls (send, mic, upload)
const startQueryInput = document.getElementById('start-query-input');
const startSendBtn = document.getElementById('start-send-btn');
const startMicBtn = document.getElementById('start-mic-btn');
const startUploadBtn = document.getElementById('start-upload-btn');
const startFileInput = document.getElementById('start-file-input');

startSendBtn?.addEventListener('click', ()=>{
    const q = (startQueryInput.value || '').trim(); if (!q) return;
    transitionToMain();
    document.getElementById('query-input').value = q;
    document.querySelector('#query-form button[type="submit"]').click();
});

// Add Enter key support for start screen
startQueryInput?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        const q = (startQueryInput.value || '').trim(); if (!q) return;
        transitionToMain();
        document.getElementById('query-input').value = q;
        document.querySelector('#query-form button[type="submit"]').click();
    }
});
startUploadBtn?.addEventListener('click', ()=> startFileInput.click());
startFileInput?.addEventListener('change', function(){
    const file = this.files && this.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev){ lastRawText = ev.target.result || ''; rawPre.textContent = lastRawText; };
    reader.readAsText(file);
    const fd = new FormData(); fd.append('file', file);
    fetch('/upload_json', { method: 'POST', body: fd })
      .then(r=>r.json()).then(data=>{
        if (data.error) return alert('JSON parse error: ' + data.error);
        const entries = data.parsed_list || (data.parsed ? [data.parsed] : []);
        if (!entries.length) return;
        entries.forEach(newEntry => {
            if (newEntry.id && !allEntries.some(existing => existing.id === newEntry.id)) {
                allEntries.push(newEntry);
            }
        });
    renderEntryList();
        transitionToMain();
        // Auto-submit first entry
        const first = allEntries[0];
        if (first && first.questionText){
            document.getElementById('query-input').value = first.questionText;
            document.querySelector('#query-form button[type="submit"]').click();
            renderEntry(first);
        }
      }).catch(err=> alert('Error during backend JSON upload: ' + err));
});

// Start screen mic uses same STT pipeline
startMicBtn?.addEventListener('click', async ()=>{
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = chooseAudioMime();
        mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        const res = await fetch('/stt/start', { method: 'POST' });
        const data0 = await res.json(); if (!res.ok || !data0.stt_sid) throw new Error(data0.error || 'Failed to start STT session');
        sttSid = data0.stt_sid; isRecording = true; startMicBtn.classList.add('recording');
        mediaRecorder.addEventListener('dataavailable', async (e) => {
            if (e.data && e.data.size > 0 && isRecording) {
                const fd = new FormData();
                const ext = extForMime(mediaRecorder.mimeType);
                const file = new File([e.data], `chunk_${Date.now()}.${ext}`, { type: mediaRecorder.mimeType });
                fd.append('audio', file);
                try { await fetch('/stt/chunk', { method: 'POST', body: fd }); } catch {}
            }
        });
        mediaRecorder.start(2000);
        const stopHandler = async ()=>{
            startMicBtn.removeEventListener('click', stopHandler);
            if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
            isRecording = false; mediaRecorder.stop(); mediaRecorder.stream.getTracks().forEach(t=>t.stop());
            startMicBtn.classList.remove('recording');
            try { const res2 = await fetch('/stt/stop', { method: 'POST' }); const data = await res2.json(); if (data.text) { startQueryInput.value = data.text; startSendBtn.click(); } } catch {}
        };
        startMicBtn.addEventListener('click', stopHandler);
    } catch (err) { alert('Unable to access microphone: ' + err.message); }
});

const sttBackend = '{{ stt_backend }}';
const micBtn = document.getElementById('mic-btn');
const queryInput = document.getElementById('query-input');
const sendBtn = document.querySelector('#query-form button[type="submit"]');

let isRecording = false;

// --- Logic for Deepgram Streaming STT ---
let socket;
let microphone;
let processor;

async function startDeepgramStreaming() {
    isRecording = true;
    micBtn.classList.add('recording');
    queryInput.placeholder = "🎤 Starting real-time conversation... Speak naturally!";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        microphone = stream;

        const wsUrl = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/stt/deepgram';
        socket = new WebSocket(wsUrl);

        socket.onopen = async () => {
            console.log('WebSocket connected for Deepgram streaming.');
            try {
                // Use MediaRecorder with a more compatible format
                let mimeType = 'audio/webm;codecs=opus';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/webm';
                }
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/ogg;codecs=opus';
                }
                
                const mediaRecorder = new MediaRecorder(microphone, { 
                    mimeType: mimeType,
                    audioBitsPerSecond: 16000 
                });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
                        console.log('Sending audio chunk, size:', event.data.size);
                        socket.send(event.data);
                    }
                };
                
                mediaRecorder.onerror = (event) => {
                    console.error('MediaRecorder error:', event.error);
                };
                
                mediaRecorder.start(250); // Send chunks every 250ms for better stability and less load
                processor = mediaRecorder; // Store reference for cleanup
                
            } catch (error) {
                console.error('Error setting up MediaRecorder:', error);
                socket.close();
            }
        };

        socket.onmessage = (event) => {
            console.log('Received WebSocket message:', event.data);
            try {
                const data = JSON.parse(event.data);
                
                if (data.transcript) {
                    // Real-time transcript updates
                    queryInput.value = data.transcript;
                    
                    // Visual feedback for interim vs final results
                    if (data.interim) {
                        queryInput.style.fontStyle = 'italic';
                        queryInput.style.opacity = '0.8';
                    } else {
                        queryInput.style.fontStyle = 'normal';
                        queryInput.style.opacity = '1';
                        // Don't add user message here - wait for LLM response
                    }
                    
                    console.log('Real-time transcript:', data.transcript, data.interim ? '(interim)' : '(final)');
                    
                } else if (data.llm_response) {
                    // LLM interjected with a response
                    console.log('LLM interjection:', data.llm_response);
                    
                    const conversation = document.getElementById('query-conversation');
                    
                    // First add the user message (the complete transcript that triggered the response)
                    if (queryInput.value.trim()) {
                        const userMessage = document.createElement('div');
                        userMessage.className = 'message-user';
                        userMessage.textContent = queryInput.value.trim();
                        conversation.appendChild(userMessage);
                    }
                    
                    // Then add LLM response
                    const botMessage = document.createElement('div');
                    botMessage.className = 'message-bot';
                    botMessage.textContent = data.llm_response;
                    conversation.appendChild(botMessage);
                    conversation.scrollTop = conversation.scrollHeight;
                    
                    // Clear the input to continue listening for new speech
                    queryInput.value = '';
                    queryInput.placeholder = '🎤 Continue speaking... AI is listening';
                    
                } else if (data.error) {
                    console.error('WebSocket error:', data.error);
                    alert('Streaming error: ' + data.error);
                } else if (data.status) {
                    console.log('WebSocket status:', data.message);
                    queryInput.placeholder = '🎤 Real-time conversation active - speak naturally!';
                }
            } catch (e) {
                console.error('Error parsing WebSocket message:', e);
            }
        };

        socket.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            if (isRecording) {
                console.log('WebSocket closed unexpectedly during recording');
                // Don't auto-reconnect, let user restart manually
            }
        };
        socket.onerror = (err) => {
            console.error('WebSocket error:', err);
            if (isRecording) {
                console.log('WebSocket error during recording, stopping...');
                stopDeepgramStreaming();
            }
        };

    } catch (err) {
        console.error('Mic/WebSocket error:', err);
        alert('Unable to access microphone or connect to streaming service: ' + err.message);
        await stopDeepgramStreaming(); // Cleanup on error
    }
}

async function stopDeepgramStreaming() {
    if (socket) socket.close();
    if (processor && processor.stop) processor.stop(); // Stop MediaRecorder
    if (microphone) microphone.getTracks().forEach(track => track.stop());
    isRecording = false;
    micBtn.classList.remove('recording');
    queryInput.placeholder = "Enter your query or use the mic...";
    
    // End the conversation session
    console.log('Ending real-time conversation session');
    queryInput.placeholder = "Real-time conversation ended. Click mic to start again.";
    
    socket = microphone = processor = null;
}

// --- Logic for chunked recording (Whisper/Gemini) ---
let mediaRecorder = null;
let sttSid = null;

function chooseAudioMime() {
    const options = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'];
    for (const t of options) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
}

function extForMime(m) {
    if (!m) return 'webm';
    if (m.includes('ogg')) return 'ogg';
    return 'webm';
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = chooseAudioMime();
        mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        
        const res = await fetch('/stt/start', { method: 'POST' });
        const data0 = await res.json();
        if (!res.ok || !data0.stt_sid) throw new Error(data0.error || 'Failed to start STT session');
        sttSid = data0.stt_sid;
        if (sendBtn) sendBtn.disabled = true;

        mediaRecorder.addEventListener('dataavailable', async (e) => {
            if (e.data && e.data.size > 0 && isRecording) {
                const fd = new FormData();
                const ext = extForMime(mediaRecorder.mimeType);
                const file = new File([e.data], `chunk_${Date.now()}.${ext}`, { type: mediaRecorder.mimeType });
                fd.append('audio', file);
                try { await fetch('/stt/chunk', { method: 'POST', body: fd }); } catch (err) { console.error('Chunk upload failed', err); }
            }
        });

        mediaRecorder.start(2000);
        isRecording = true;
        micBtn.classList.add('recording');
        queryInput.placeholder = "Recording... Click mic to stop.";
    } catch (err) {
        console.error('Mic error:', err);
        alert('Unable to access microphone: ' + err.message);
    }
}

async function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
    isRecording = false;
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    micBtn.classList.remove('recording');
    if (sendBtn) sendBtn.disabled = false;
    queryInput.placeholder = "Finalizing transcription...";

    try {
        const res = await fetch('/stt/stop', { method: 'POST' });
        const data = await res.json();
        if (data.text) {
            queryInput.value = data.text;
            sendBtn.click();
        } else if (data.error) {
            console.error('STT error:', data.error);
            alert('Transcription failed: ' + data.error);
        }
    } catch (e) {
        console.error('Failed to stop STT session:', e);
        alert('An error occurred while finalizing transcription.');
    } finally {
        queryInput.placeholder = "Enter your query or use the mic...";
        mediaRecorder = null;
        sttSid = null;
    }
}

// Main mic button event listener
micBtn?.addEventListener('click', async () => {
    if (sttBackend === 'deepgram_streaming') {
        if (isRecording) await stopDeepgramStreaming(); else await startDeepgramStreaming();
    } else {
        if (isRecording) await stopRecording(); else await startRecording();
    }
});

document.getElementById('upload-form')?.addEventListener('submit', function(e) {
        e.preventDefault();

        const fileInput = document.getElementById('file-input');
        const file = fileInput.files[0];

        if (!file) {
            alert("Please select a file.");
            return;
        }

        //display raw json (modal)
        const reader = new FileReader();
        reader.onload = function(event) {
            const rawText = event.target.result; lastRawText = rawText || '';
            rawPre.textContent = rawText;
        };
        reader.onerror = function(event) {
            alert("Error reading: " + event.target.error.name);
        };
        reader.readAsText(file);

        //send to backend for JSON parsing and UI hydration
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload_json', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('JSON parse error:', data.error);
                alert('JSON parse error: ' + data.error);
                return;
            }
            console.log(data.message);
            const entries = data.parsed_list || (data.parsed ? [data.parsed] : []);
            if (!entries.length) return;

            // Merge new entries, ensuring no duplicates based on ID
            let newEntriesAdded = false;
            entries.forEach(newEntry => {
                if (newEntry.id && !allEntries.some(existing => existing.id === newEntry.id)) {
                    allEntries.push(newEntry);
                    newEntriesAdded = true;
                }
            });

            renderEntryList();

            // If this is the first upload or new entries were added,
            // render the details of the first overall entry.
            if (allEntries.length > 0 && newEntriesAdded) {
                renderEntry(allEntries[0]);
                // Highlight the first entry button as active
                const firstBtn = document.querySelector('.entry-btn');
                if(firstBtn) firstBtn.classList.add('active');
            }
        })
        .catch(error => {
            console.error('Error during backend JSON upload:', error);
        });
    });

document.getElementById('query-form').addEventListener('submit', function(e) {
        e.preventDefault();

        const queryInput = document.getElementById('query-input');
        const query = queryInput.value.trim();
        if (!query) return;

        // Display user message
        const conversation = document.getElementById('query-conversation');
        const userMessage = document.createElement('div');
        userMessage.className = 'message-user';
        userMessage.textContent = query;
        conversation.appendChild(userMessage);
        conversation.scrollTop = conversation.scrollHeight;

        // Log turn
        conversationLog.push({
            turn: conversationLog.length + 1,
            role: 'user',
            inputType: 'text',
            content: query,
            timestamp: new Date().toISOString()
        });

        // Clear input
        queryInput.value = '';

        //send query to backend
        const queryData = new FormData();
        queryData.append('query', query);

        fetch ('/query', {
            method: 'POST',
            body: queryData
        })
        .then(response => response.json())
        .then(data => {
            // Display bot response
            const botMessage = document.createElement('div');
            botMessage.className = 'message-bot';
            botMessage.textContent = data.response || (data.error ? ('Error: ' + data.error) : '');
            conversation.appendChild(botMessage);
            conversation.scrollTop = conversation.scrollHeight;

            // Log bot turn
            conversationLog.push({
                turn: conversationLog.length + 1,
                role: 'assistant',
                inputType: 'model',
                content: botMessage.textContent,
                timestamp: new Date().toISOString()
            });
        })
        .catch(error => {
            console.error('Error during backend query:', error);
        });
    });

// Save chat button handler
function renderEntryList() {
    const entryListContainer = document.getElementById('entry-list-container');
    const entryList = document.getElementById('entry-list');
    entryList.innerHTML = '';

    if (allEntries.length > 1) {
        entryListContainer.classList.remove('hidden');
    } else {
        entryListContainer.classList.add('hidden');
    }

    allEntries.forEach((entry, index) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'entry-item-wrapper';

        const btn = document.createElement('button');
        btn.textContent = entry.id || `Entry ${index + 1}`;
        btn.className = 'btn entry-btn';
        btn.dataset.entryId = entry.id;
        btn.addEventListener('click', () => {
            renderEntry(entry);
            document.querySelectorAll('.entry-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });

        const removeBtn = document.createElement('button');
        removeBtn.textContent = '×';
        removeBtn.className = 'btn-remove-entry';
        removeBtn.title = 'Remove entry';
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent the main button click
            allEntries = allEntries.filter(e => e.id !== entry.id);
            renderEntryList();
            // Optional: Clear the view if the active entry was removed
            if (lastParsed && lastParsed.id === entry.id) {
                document.getElementById('json-meta').innerHTML = '';
                document.getElementById('question-display').innerHTML = '';
                rawPre.textContent = '';
                lastParsed = null;
            }
        });

        wrapper.appendChild(btn);
        wrapper.appendChild(removeBtn);
        entryList.appendChild(wrapper);
    });
}

document.getElementById('save-chat-btn')?.addEventListener('click', function() {
    const payload = {
        sessionInfo: {
            sessionId: crypto.randomUUID(),
            userId: 'anonymous',
            startTimestamp: sessionStart,
            endTimestamp: new Date().toISOString(),
            llmModel: 'gemini-1.5-flash-latest'
        },
        context: {
            initialContent: lastParsed?.questionText || '',
            finalContent: conversationLog.length ? conversationLog[conversationLog.length - 1].content : ''
        },
        conversationLog: conversationLog,
        evaluation: {
            surveyResponses: {},
            userComments: ''
        }
    };
    fetch('/save_chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(d => {
        if (d.error) {
            alert('Failed to save: ' + d.error);
        } else {
            alert('Saved: ' + d.dirName);
        }
    })
    .catch(err => {
        alert('Save error: ' + err);
    });
});

function renderEntry(entry) {
    lastParsed = entry;

    // Clear previous meta content
    document.getElementById('json-meta').innerHTML = '';

    // Populate the new question display area at the bottom
    const questionDiv = document.getElementById('question-display');
    questionDiv.innerHTML = ''; // Clear previous question
    const qP = document.createElement('p');
    qP.innerHTML = `<strong>Question:</strong><br>${entry.questionText || ''}`;
    questionDiv.appendChild(qP);

    const submitBtn = document.createElement('button');
    submitBtn.textContent = 'Submit Question to Chat';
    submitBtn.className = 'btn submit-question-btn';
    submitBtn.addEventListener('click', () => {
        const q = (entry.questionText || '').trim();
        if (q) {
            document.getElementById('query-input').value = q;
            document.querySelector('#query-form button[type="submit"]').click();
        }
    });
    questionDiv.appendChild(submitBtn);

    // Update raw JSON view in modal if raw present
    if (entry.raw) { rawPre.textContent = JSON.stringify(entry.raw, null, 2); }
}

const submitAction = async (action) => {
  const formData = new FormData();
  formData.append('action', action);
  const res = await fetch('/step', { method: 'POST', body: formData });
  const data = await res.json();
  if (data.turn) {
    const conversation = document.getElementById('query-conversation');
    const userMessage = document.createElement('div');
    userMessage.className = 'message-user';
    userMessage.textContent = `Action: ${data.turn.action}`;
    conversation.appendChild(userMessage);
    conversation.scrollTop = conversation.scrollHeight;
  }
  renderAvailableActions(data.snapshot.actions);
};

function renderAvailableActions(actions){
const actionsDiv = document.getElementById('actions');
if(actionsDiv) {
actionsDiv.innerHTML = '';
(actions || []).forEach(a => {
const b = document.createElement('button');
b.className = 'action-btn';
b.dataset.action = a;
b.textContent = a;
b.onclick = () => submitAction(a);
actionsDiv.appendChild(b);
});
}
}

document.addEventListener('DOMContentLoaded', () => {
const actionsDiv = document.getElementById('actions');
if(actionsDiv) {
const actionsData = actionsDiv.dataset.actions;
if (actionsData) {
try {
const snapshot_actions = JSON.parse(actionsData);
renderAvailableActions(snapshot_actions);
} catch (e) {
console.error('Failed to parse actions data:', e);
}
}
}
});

