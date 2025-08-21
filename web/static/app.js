const log = (msg) => {
  const el = document.getElementById('log');
  el.textContent += `${msg}\n`;
};

const refreshFrom = (data) => {
  document.getElementById('state').textContent = data.snapshot.render;
  const actionsDiv = document.getElementById('actions');
  actionsDiv.innerHTML = '';
  (data.snapshot.actions || []).forEach(a => {
    const b = document.createElement('button');
    b.className = 'action-btn';
    b.dataset.action = a;
    b.textContent = a;
    b.onclick = () => submitAction(a);
    actionsDiv.appendChild(b);
  });
};

const submitAction = async (action) => {
  const formData = new FormData();
  formData.append('action', action);
  const res = await fetch('/step', { method: 'POST', body: formData });
  const data = await res.json();
  if (data.turn) log(`You -> ${data.turn.action} (${data.turn.status})`);
  refreshFrom(data);
};

const humanForm = document.getElementById('action-form');
humanForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const inp = document.getElementById('action');
  await submitAction(inp.value.trim());
  inp.value = '';
});

document.getElementById('model-step')?.addEventListener('click', async () => {
  const formData = new FormData();
  const res = await fetch('/step', { method: 'POST', body: formData });
  const data = await res.json();
  if (data.turn) log(`Model -> ${data.turn.action} (${data.turn.status})`);
  refreshFrom(data);
});
