'use client';
import React from 'react';
import { apiAsk } from '@/lib/api';

type Message = { role: 'user' | 'assistant', content: string, citations?: {title:string, section?:string}[], chunks?: {title:string, section?:string, text:string}[] };

export default function Chat() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [q, setQ] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const send = async () => {
    if (!q.trim()) return;
    const my = { role: 'user' as const, content: q };
    setMessages(m => [...m, my]);
    setLoading(true);
    try {
      const res = await apiAsk(q);
      const ai: Message = { role: 'assistant', content: res.answer, citations: res.citations, chunks: res.chunks };
      setMessages(m => [...m, ai]);
    } catch (e:any) {
      setMessages(m => [...m, { role: 'assistant', content: 'Error: ' + e.message }]);
    } finally {
      setLoading(false);
      setQ('');
    }
  };

  return (
    <div className="card">
      <h2>Chat</h2>
      <div style={{maxHeight: 320, overflowY:'auto', padding: 8, border:'1px solid #eee', borderRadius: 8, marginBottom: 12}}>
        {messages.map((m, i) => (
          <div key={i} style={{margin: '8px 0'}}>
            <div style={{fontSize:12, color:'#666'}}>{m.role === 'user' ? 'You' : 'Assistant'}</div>
            <div>{m.content}</div>
            {m.citations && m.citations.length>0 && (
              <div style={{marginTop:6}}>
                {m.citations.map((c, idx) => (
                  <span key={idx} className="badge" title={c.section || ''}>{c.title}</span>
                ))}
              </div>
            )}
            {m.chunks && m.chunks.length>0 && (
              <details style={{marginTop:6}}>
                <summary>View supporting chunks</summary>
                {m.chunks.map((c, idx) => (
                  <div key={idx} className="card">
                    <div style={{fontWeight:600}}>{c.title}{c.section ? ' — ' + c.section : ''}</div>
                    <div style={{whiteSpace:'pre-wrap'}}>{c.text}</div>
                  </div>
                ))}
              </details>
            )}
          </div>
        ))}
      </div>
      <div style={{display:'flex', gap:8}}>
        <input placeholder="Ask about policy or products..." value={q} onChange={e=>setQ(e.target.value)} style={{flex:1, padding:10, borderRadius:8, border:'1px solid #ddd'}} onKeyDown={(e)=>{ if(e.key==='Enter') send(); }}/>
        <button onClick={send} disabled={loading} style={{padding:'10px 14px', borderRadius:8, border:'1px solid #111', background:'#111', color:'#fff'}}>
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </div>
    </div>
  );
}
