/* global Excel, Office */
let ws: WebSocket | null = null;
let bufferText = "";
let autoSearchThreshold = 6;
const HIGHLIGHT_COLOR = "yellow";

function setStatus(msg: string) {
  const el = document.getElementById("status")!;
  el.textContent = msg;
}

function setLive(text: string) {
  const el = document.getElementById("live")!;
  el.textContent = text;
}

async function clearHighlights() {
  await Excel.run(async (ctx) => {
    const sheet = ctx.workbook.worksheets.getActiveWorksheet();
    const used = sheet.getUsedRange();
    used.load("format/fill/color");
    await ctx.sync();
    // Reset fill on used range
    used.format.fill.clear();
    await ctx.sync();
  });
}

async function highlightMatches(needle: string) {
  if (!needle || !needle.trim()) return;
  const query = needle.trim().toLowerCase();

  await Excel.run(async (ctx) => {
    const sheet = ctx.workbook.worksheets.getActiveWorksheet();
    const used = sheet.getUsedRange();
    used.load("values, rowCount, columnCount");
    await ctx.sync();

    const hits: Array<{ r: number; c: number }> = [];
    const vals = used.values as any[][];
    for (let r = 0; r < used.rowCount; r++) {
      for (let c = 0; c < used.columnCount; c++) {
        const cellVal = vals[r][c];
        if (cellVal != null && String(cellVal).toLowerCase().includes(query)) {
          hits.push({ r, c });
        }
      }
    }

    // Clear old highlights first
    used.format.fill.clear();

    // Highlight matches
    hits.forEach(({ r, c }) => {
      const cell = used.getCell(r, c);
      cell.format.fill.color = HIGHLIGHT_COLOR;
    });

    // If there are hits, select the first one
    if (hits.length > 0) {
      used.getCell(hits[0].r, hits[0].c).select();
      setStatus(`Found ${hits.length} match(es) for “${needle}”.`);
    } else {
      setStatus(`No matches for “${needle}”.`);
    }

    await ctx.sync();
  });
}

function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  ws = new WebSocket("wss://localhost:8000/ws/transcribe");

  ws.onopen = () => {
    setStatus("Connected to local STT server.");
    (document.getElementById("connect") as HTMLButtonElement).disabled = true;
    (document.getElementById("disconnect") as HTMLButtonElement).disabled = false;
  };

  ws.onmessage = async (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "partial" && typeof msg.text === "string") {
        bufferText = msg.text;
        setLive(bufferText);

        // Auto-search when transcript is "confident enough" (simple heuristic: length)
        if (bufferText.length >= autoSearchThreshold) {
          // debounce a bit so we don't spam Excel
          await highlightMatches(bufferText);
        }
      }
    } catch (e) {
      // ignore malformed messages
    }
  };

  ws.onclose = () => {
    setStatus("Disconnected.");
    (document.getElementById("connect") as HTMLButtonElement).disabled = false;
    (document.getElementById("disconnect") as HTMLButtonElement).disabled = true;
  };

  ws.onerror = () => {
    setStatus("Unable to connect. Make sure the Python server is running on :8000.");
  };
}

function disconnect() {
  if (ws) {
    ws.close();
    ws = null;
  }
}

Office.onReady(() => {
  document.getElementById("connect")!.addEventListener("click", connect);
  document.getElementById("disconnect")!.addEventListener("click", disconnect);
  document.getElementById("searchNow")!.addEventListener("click", () => highlightMatches(bufferText));
  document.getElementById("clearHighlights")!.addEventListener("click", () => clearHighlights());
  const th = document.getElementById("threshold") as HTMLInputElement;
  th.addEventListener("change", () => (autoSearchThreshold = Math.max(0, Number(th.value))));
});
