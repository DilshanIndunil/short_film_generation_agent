import React, { memo, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import {
  Sparkles,
  BrainCircuit,
  RefreshCw,
  ExternalLink,
  Play,
  Loader2,
  Image as ImageIcon,
  Users,
  Cpu,
  Download,
  Wand2,
  Film,
  Volume2,
  Copy,
  Check,
  PlusCircle
} from 'lucide-react';

// ============================================================================
// NODE 0: Image Generation Model Selector Card (Before Scenario Input)
// ============================================================================
const IMAGE_MODEL_OPTIONS = [
  {
    id: 'Qwen-Image-2.1',
    title: 'Qwen-Image-2.1 (models/qwen)',
    badge: 'LOCAL QWEN 7B',
    badgeColor: '#38bdf8',
    path: 'backend/models/qwen',
    desc: 'Unified Qwen3-VL + 7B Single-Stream DiT model in models/qwen. Realistic textures, lighting & multi-reference character fidelity.',
  },
  {
    id: 'FLUX.1-dev',
    title: 'FLUX.1-dev + Character LoRA',
    badge: 'FINE-TUNED LORA',
    badgeColor: '#a855f7',
    path: 'backend/models/film_characters_lora.safetensors',
    desc: 'Cinema Diffusion + Custom Fine-Tuned arjun_male & maya_female 360° Turnaround LoRA weights.',
  },
  {
    id: 'FLUX.1-schnell',
    title: 'FLUX.1-schnell (FP8 Local)',
    badge: 'FAST 4-STEP FP8',
    badgeColor: '#34d399',
    path: 'backend/models/flux1-schnell-fp8.safetensors',
    desc: 'High-speed 4-step FP8 local GPU inference with sequential CPU offload for 8GB VRAM.',
  },
  {
    id: 'FLUX.2',
    title: 'FLUX.2 Pro Studio',
    badge: 'PRO STUDIO',
    badgeColor: '#f59e0b',
    path: 'backend/models/film_characters_lora.safetensors',
    desc: 'Maximum dynamic range & anamorphic cinema grading with Character Consistency LoRA.',
  },
];

export const VIDEO_MODEL_OPTIONS = [
  {
    id: 'Wan2.2-I2V-A14B',
    title: 'Wan2.2-I2V-A14B (Colab A100 / Drive AI_Models)',
    badge: 'WAN 2.2 14B I2V',
    badgeColor: '#10b981',
    path: 'Dilshan/AI_Models/Wan2.2-I2V-A14B-Diffusers',
    desc: 'Primary 14B MoE Image-to-Video Cinema Engine on Google Colab GPU for maximum facial & lip realism.',
  },
  {
    id: 'Wan2.2-T2V-A14B',
    title: 'Wan2.2-T2V-A14B (Colab A100 / Drive AI_Models)',
    badge: 'WAN 2.2 14B T2V',
    badgeColor: '#f59e0b',
    path: 'Dilshan/AI_Models/Wan2.2-T2V-A14B-Diffusers',
    desc: 'Full Wan 2.2 A14B Text-to-Video Diffusers pipeline in Google Drive AI_Models.',
  },
  {
    id: 'LTX-Video-13B',
    title: 'LTX-2.5 / LTX-Video-13B (Drive AI_Models / Local)',
    badge: 'LTX 2.5 / 13B',
    badgeColor: '#ec4899',
    path: 'Dilshan/AI_Models/LTX-2.5',
    desc: 'High-detail LTX-2.5 / 13B Distilled FP8 Video checkpoint with spatial & temporal upscalers.',
  },
  {
    id: 'LTX-Video-2B',
    title: 'LTX-Video-2B Distilled (Local Fallback)',
    badge: 'LOCAL LTX 2B',
    badgeColor: '#f43f5e',
    path: 'backend/models/models/ltx',
    desc: 'Lightricks LTX-Video 2B v0.9.8 Distilled in models/models/ltx for local GPU fallback.',
  },
];

export const NodeModelSelector = memo(({ data, isConnectable }) => {
  const selectedModel = data.fluxModel || 'Qwen-Image-2.1';
  const selectedVideoModel = data.wanModel || 'Wan2.2-I2V-A14B';
  const aspectRatio = data.aspectRatio || '16:9';
  const activeOption = IMAGE_MODEL_OPTIONS.find((m) => m.id === selectedModel) || IMAGE_MODEL_OPTIONS[0];
  const activeVideoOption = VIDEO_MODEL_OPTIONS.find((m) => m.id === selectedVideoModel) || VIDEO_MODEL_OPTIONS[0];
  const [colabInput, setColabInput] = useState(data.remoteGpuUrl || '');
  const [isTestingColab, setIsTestingColab] = useState(false);

  React.useEffect(() => {
    if (data.remoteGpuUrl !== undefined && data.remoteGpuUrl !== colabInput) {
      setColabInput(data.remoteGpuUrl || '');
    }
  }, [data.remoteGpuUrl]);

  const handleSaveColabUrl = async (urlToSave) => {
    if (!data.onUpdateRemoteGpuUrl) return;
    setIsTestingColab(true);
    try {
      await data.onUpdateRemoteGpuUrl(urlToSave);
    } finally {
      setIsTestingColab(false);
    }
  };

  return (
    <div className={`studio-node model-selector-node accent-sky ${data.isActive ? 'active-processing' : ''}`}>
      <div className="studio-node-header">
        <div className="studio-node-title">
          <Cpu size={16} style={{ color: '#38bdf8' }} />
          <span>0. Image & Video Models</span>
        </div>
        <span
          className="studio-node-badge"
          style={{ color: '#38bdf8', borderColor: 'rgba(56, 189, 248, 0.4)' }}
        >
          Select Engines
        </span>
      </div>

      <div className="studio-node-body">
        <div style={{ fontSize: '0.69rem', color: '#38bdf8', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
          🖼️ 1. Image Generation Model (Card 3.K Master Frame):
        </div>

        {/* Image Model Option Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.38rem' }}>
          {IMAGE_MODEL_OPTIONS.map((opt) => {
            const isSelected = selectedModel === opt.id;
            return (
              <div
                key={opt.id}
                className={`model-option-card nodrag ${isSelected ? 'selected' : ''}`}
                onClick={() => data.onSelectFluxModel && data.onSelectFluxModel(opt.id)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span
                      style={{
                        width: '9px',
                        height: '9px',
                        borderRadius: '50%',
                        background: isSelected ? '#38bdf8' : 'rgba(255,255,255,0.2)',
                        boxShadow: isSelected ? '0 0 8px #38bdf8' : 'none',
                      }}
                    />
                    <span style={{ fontWeight: 700, fontSize: '0.73rem', color: '#ffffff' }}>
                      {opt.title}
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: '0.56rem',
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      background: 'rgba(255,255,255,0.06)',
                      color: opt.badgeColor,
                      border: `1px solid ${opt.badgeColor}55`,
                    }}
                  >
                    {opt.badge}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2px', fontSize: '0.58rem', fontFamily: 'var(--font-mono)', color: isSelected ? '#7dd3fc' : '#64748b' }}>
                  <span>📂 {opt.path}</span>
                  <span>{isSelected ? '✓ ACTIVE' : 'Click to use'}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Video Generation Model Selector (LTX & Wan 2.2) */}
        <div style={{ paddingTop: '0.45rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.38rem' }}>
          <div style={{ fontSize: '0.69rem', color: '#fda4af', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
            🎬 2. Video Generation Model (Card 4.K — LTX / Wan 2.2):
          </div>

          {VIDEO_MODEL_OPTIONS.map((vopt) => {
            const isVidSelected = selectedVideoModel === vopt.id;
            return (
              <div
                key={vopt.id}
                className={`model-option-card nodrag ${isVidSelected ? 'selected' : ''}`}
                style={
                  isVidSelected
                    ? { borderColor: 'rgba(244, 63, 94, 0.7)', background: 'rgba(244, 63, 94, 0.11)' }
                    : {}
                }
                onClick={() => data.onSelectWanModel && data.onSelectWanModel(vopt.id)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span
                      style={{
                        width: '9px',
                        height: '9px',
                        borderRadius: '50%',
                        background: isVidSelected ? '#f43f5e' : 'rgba(255,255,255,0.2)',
                        boxShadow: isVidSelected ? '0 0 8px #f43f5e' : 'none',
                      }}
                    />
                    <span style={{ fontWeight: 700, fontSize: '0.73rem', color: '#ffffff' }}>
                      {vopt.title}
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: '0.56rem',
                      fontWeight: 700,
                      padding: '2px 6px',
                      borderRadius: '4px',
                      background: 'rgba(255,255,255,0.06)',
                      color: vopt.badgeColor,
                      border: `1px solid ${vopt.badgeColor}55`,
                    }}
                  >
                    {vopt.badge}
                  </span>
                </div>

                <div style={{ fontSize: '0.62rem', color: '#94a3b8', lineHeight: 1.3 }}>
                  {vopt.desc}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2px', fontSize: '0.58rem', fontFamily: 'var(--font-mono)', color: isVidSelected ? '#fda4af' : '#64748b' }}>
                  <span>📂 {vopt.path}</span>
                  <span>{isVidSelected ? '✓ ACTIVE' : 'Click to use'}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Aspect Ratio Selector */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '0.35rem', borderTop: '1px solid var(--border-subtle)' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600 }}>Frame Format (Ratio):</span>
          <div className="selector-chip-group">
            {['16:9', '9:16', '1:1'].map((ratio) => (
              <button
                key={ratio}
                type="button"
                className={`selector-chip nodrag ${aspectRatio === ratio ? 'active' : ''}`}
                onClick={() => data.onSelectRatio && data.onSelectRatio(ratio)}
              >
                {ratio}
              </button>
            ))}
          </div>
        </div>

        {/* Google Colab Remote GPU Tunnel URL (video_gen_colab_gpu_studio.ipynb) */}
        <div style={{ paddingTop: '0.4rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.32rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.67rem', color: '#34d399', fontWeight: 700, textTransform: 'uppercase' }}>
              ☁️ 3. Google Colab Video GPU URL (Optional):
            </span>
            <span
              style={{
                fontSize: '0.56rem',
                fontWeight: 700,
                padding: '1px 6px',
                borderRadius: '4px',
                background: data.remoteGpuOnline ? 'rgba(16, 185, 129, 0.16)' : 'rgba(148, 163, 184, 0.14)',
                color: data.remoteGpuOnline ? '#34d399' : '#94a3b8',
                border: `1px solid ${data.remoteGpuOnline ? 'rgba(16, 185, 129, 0.45)' : 'rgba(148, 163, 184, 0.3)'}`,
              }}
            >
              {data.remoteGpuOnline ? `🟢 COLAB ONLINE (${data.remoteGpuDevice || 'GPU'})` : '⚪ LOCAL GPU MODE'}
            </span>
          </div>

          <div style={{ display: 'flex', gap: '0.3rem', alignItems: 'center' }}>
            <input
              type="text"
              className="nodrag"
              value={colabInput}
              onChange={(e) => setColabInput(e.target.value)}
              placeholder="https://xxxx.trycloudflare.com"
              style={{
                flex: 1,
                fontSize: '0.64rem',
                padding: '5px 8px',
                borderRadius: '6px',
                border: '1px solid rgba(56, 189, 248, 0.35)',
                background: 'rgba(15, 23, 42, 0.85)',
                color: '#f8fafc',
                fontFamily: 'var(--font-mono)',
              }}
            />
            <button
              type="button"
              className="selector-chip nodrag"
              disabled={isTestingColab}
              onClick={() => handleSaveColabUrl(colabInput.trim())}
              style={{
                fontSize: '0.62rem',
                padding: '5px 9px',
                background: 'rgba(16, 185, 129, 0.2)',
                borderColor: 'rgba(16, 185, 129, 0.55)',
                color: '#6ee7b7',
                fontWeight: 700,
              }}
            >
              {isTestingColab ? '...' : 'Connect'}
            </button>
            {colabInput && (
              <button
                type="button"
                className="selector-chip nodrag"
                disabled={isTestingColab}
                onClick={() => {
                  setColabInput('');
                  handleSaveColabUrl('');
                }}
                style={{ fontSize: '0.6rem', padding: '5px 7px', color: '#fda4af' }}
                title="Clear Colab URL & use Local GPU"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        {/* Active Hardware & Model Summary Box */}
        <div style={{ background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.3)', borderRadius: '7px', padding: '0.45rem 0.6rem', display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '0.64rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#cbd2de' }}>
              Image Engine: <strong style={{ color: '#38bdf8' }}>{activeOption.id}</strong>
            </span>
            <span style={{ color: '#34d399', fontWeight: 700 }}>➔ Card 3.K</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: '#cbd2de' }}>
              Video Engine:{' '}
              <strong style={{ color: '#fda4af' }}>
                {data.remoteGpuOnline ? `${data.remoteGpuEngine || activeVideoOption.id} (Colab)` : activeVideoOption.id}
              </strong>
            </span>
            <span style={{ color: '#fda4af', fontWeight: 700 }}>➔ Card 4.K</span>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Right} isConnectable={isConnectable} />
    </div>
  );
});

// ============================================================================
// NODE 1: Speaker Character & Spoken Script Input Card
// ============================================================================
export const segmentScriptIntoBeats = (storyText = '') => {
  const cleaned = String(storyText || '').trim();
  if (!cleaned) return [];
  const rawChunks = cleaned
    .split(/\n+/)
    .map((p) => p.trim())
    .filter(Boolean);
  const sentenceBeats = [];
  for (const chunk of rawChunks) {
    const parts = chunk
      .split(/(?<=[.!?।])\s+|;\s+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 2);
    sentenceBeats.push(...parts);
  }
  const beats = [];
  for (const sent of sentenceBeats) {
    const wordCount = sent.split(/\s+/).filter(Boolean).length;
    if (wordCount > 26 || sent.length > 165 || (sentenceBeats.length === 1 && wordCount > 14)) {
      const subParts = sent
        .split(/,\s+(?=[A-Za-z\u0D80-\u0DFF])|\s+—\s+|\s+-\s+/)
        .map((c) => c.trim())
        .filter((c) => c.split(/\s+/).length >= 3);
      if (subParts.length > 1) {
        beats.push(...subParts);
      } else {
        beats.push(sent);
      }
    } else {
      beats.push(sent);
    }
  }
  if (beats.length === 0 && cleaned) {
    beats.push(cleaned);
  }
  return beats.slice(0, 12);
};

export const NodePrompt = memo(({ data, isConnectable }) => {
  const scenes = data.scenes || [{ id: 1, text: '', duration: '5s' }];
  const storyText =
    data.storyText !== undefined
      ? data.storyText
      : scenes
        .map((s) => s.text)
        .filter(Boolean)
        .join('\n\n');
  const selectedChars = data.characters || ['female'];
  const fluxModel = data.fluxModel || 'Qwen-Image-2.1';
  const liveBeatsCount = segmentScriptIntoBeats(storyText).length;
  const extractedCount = data.extractedSceneCount > 0 ? data.extractedSceneCount : liveBeatsCount;

  const hasFemale = selectedChars.includes('female');
  const hasMale = selectedChars.includes('male') && !hasFemale;
  const activeSpeakerLabel = hasFemale ? 'Girl (Maya — Female Lead)' : 'Man (Arjun — Male Lead)';

  return (
    <div className={`studio-node prompt-node accent-purple ${data.isActive ? 'active-processing' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />
      <div className="studio-node-header">
        <div className="studio-node-title">
          <Sparkles size={16} style={{ color: '#a855f7' }} />
          <span>1. Character & Spoken Script Input</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          {data.onNewGeneration && (
            <button
              type="button"
              className="planner-reopt-all-btn nodrag"
              style={{
                background: 'rgba(168, 85, 247, 0.18)',
                borderColor: 'rgba(168, 85, 247, 0.55)',
                color: '#e9d5ff',
              }}
              onClick={(e) => {
                e.stopPropagation();
                data.onNewGeneration();
              }}
              disabled={data.isRunning || data.isOptimizing}
              title="Clear state and start a fresh New Generation (New Master Frame + New Script)"
            >
              <PlusCircle size={11} />
              <span>New Generation</span>
            </button>
          )}
          <span className="studio-node-badge" style={{ color: '#d8b4fe', borderColor: 'rgba(168, 85, 247, 0.4)' }}>
            {fluxModel}
          </span>
        </div>
      </div>

      <div className="studio-node-body">
        {/* Active Model & Format Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', paddingBottom: '0.45rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.71rem', color: 'var(--text-dim)', fontWeight: 600 }}>Image Model (Card 0):</span>
            <span style={{ fontSize: '0.68rem', color: '#38bdf8', fontWeight: 700, background: 'rgba(56, 189, 248, 0.12)', padding: '2px 8px', borderRadius: '5px', border: '1px solid rgba(56, 189, 248, 0.35)' }}>
              ⚡ {fluxModel}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.71rem', color: 'var(--text-dim)', fontWeight: 600 }}>Master Frame & Script:</span>
            <span
              style={{
                fontSize: '0.65rem',
                color: extractedCount > 0 ? '#34d399' : '#c084fc',
                fontWeight: 700,
                background: extractedCount > 0 ? 'rgba(16, 185, 129, 0.14)' : 'rgba(168, 85, 247, 0.14)',
                padding: '2px 8px',
                borderRadius: '5px',
                border: `1px solid ${extractedCount > 0 ? 'rgba(16, 185, 129, 0.4)' : 'rgba(168, 85, 247, 0.38)'}`,
              }}
            >
              {extractedCount > 0
                ? `✨ 1 New Master Frame • ${extractedCount} Script Scene(s)`
                : '🎬 1 Fresh Master Frame + Auto Script Breakdown'}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.71rem', color: 'var(--text-dim)', fontWeight: 600 }}>Format (Ratio):</span>
            <div className="selector-chip-group">
              {['16:9', '9:16', '1:1'].map((ratio) => (
                <button
                  key={ratio}
                  type="button"
                  className={`selector-chip nodrag ${data.aspectRatio === ratio ? 'active' : ''}`}
                  onClick={() => data.onSelectRatio && data.onSelectRatio(ratio)}
                >
                  {ratio}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Speaker Character Selector (Girl OR Man) */}
        <div style={{ paddingBottom: '0.45rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Users size={12} style={{ color: '#a855f7' }} />
              Select Speaker Character (Girl or Man):
            </span>
            <span style={{ fontSize: '0.62rem', color: '#d8b4fe', fontWeight: 700 }}>
              {activeSpeakerLabel}
            </span>
          </div>

          <div className="char-ref-strip">
            <div
              className={`char-ref-card nodrag ${hasFemale ? 'selected' : ''}`}
              onClick={() => data.onToggleCharacter && data.onToggleCharacter('female')}
              title="Select Female Lead (Girl — Maya) as the Speaker"
            >
              <img
                src="/characters/female_turnaround.png"
                onError={(e) => {
                  if (!e.currentTarget.src.includes('/data/characters/')) {
                    e.currentTarget.src = '/data/characters/female_turnaround.png';
                  }
                }}
                alt="Female Lead (Girl)"
                className="char-ref-thumb"
              />
              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#fff' }}>Female Lead (Girl)</div>
                <div style={{ fontSize: '0.6rem', color: hasFemale ? '#d8b4fe' : 'var(--text-dim)' }}>
                  {hasFemale ? '✓ Selected Speaker' : 'Click to select Girl'}
                </div>
              </div>
            </div>

            <div
              className={`char-ref-card nodrag ${hasMale ? 'selected' : ''}`}
              onClick={() => data.onToggleCharacter && data.onToggleCharacter('male')}
              title="Select Male Lead (Man — Arjun) as the Speaker"
            >
              <img
                src="/characters/male_turnaround.png"
                onError={(e) => {
                  if (!e.currentTarget.src.includes('/data/characters/')) {
                    e.currentTarget.src = '/data/characters/male_turnaround.png';
                  }
                }}
                alt="Male Lead (Man)"
                className="char-ref-thumb"
              />
              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#fff' }}>Male Lead (Man)</div>
                <div style={{ fontSize: '0.6rem', color: hasMale ? '#d8b4fe' : 'var(--text-dim)' }}>
                  {hasMale ? '✓ Selected Speaker' : 'Click to select Man'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Spoken Script Input Box */}
        <div className="scenes-scroll-list">
          <div className="scene-card-item">
            <div className="scene-card-header">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <span className="scene-num-badge">SCRIPT</span>
                  <span style={{ fontWeight: 700, color: 'var(--text-main)', fontSize: '0.72rem' }}>
                    {hasFemale ? 'Girl (Maya) Spoken Script' : 'Man (Arjun) Spoken Script'}
                  </span>
                </div>
                <span style={{ fontSize: '0.61rem', color: '#a855f7', fontWeight: 600 }}>
                  1 Master Frame • Auto Script Scenes
                </span>
              </div>
            </div>

            <textarea
              className="node-textarea nodrag nowheel"
              style={{ minHeight: '138px', lineHeight: 1.5, fontSize: '0.74rem' }}
              value={storyText}
              onChange={(e) => data.onChangeStoryText && data.onChangeStoryText(e.target.value)}
              onMouseDown={(e) => e.stopPropagation()}
              placeholder={`Enter ONLY the script spoken by the ${hasFemale ? 'Girl (Maya)' : 'Man (Arjun)'} (in English, Sinhala, or Singlish).\n\nExample: "Success never comes overnight; it is built in the quiet hours when nobody is watching. Every small step you take today shapes the person you become tomorrow. Keep your mindset strong and never stop believing in your journey."\n\n• Every generation creates a BRAND-NEW Studio Master Frame Image!\n• Your script is automatically broken down into Scene 1..N!`}
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ marginTop: '0.3rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
          <button
            type="button"
            className="card-optimize-btn nodrag"
            onClick={(e) => {
              e.stopPropagation();
              if (data.onOptimizeOnly) data.onOptimizeOnly();
            }}
            disabled={data.isRunning || data.isOptimizing}
          >
            {data.isOptimizing ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                <span>Breaking Down Script & Creating Fresh Master Prompt...</span>
              </>
            ) : (
              <>
                <Wand2 size={14} />
                <span>Step 1: Breakdown Script & New Master Prompt ➔ Card 2</span>
              </>
            )}
          </button>

          <button
            type="button"
            className="card-generate-btn nodrag"
            onClick={(e) => {
              e.stopPropagation();
              if (data.onRunWorkflow) data.onRunWorkflow();
            }}
            disabled={data.isRunning || data.isOptimizing}
          >
            {data.isRunning ? (
              <>
                <Loader2 size={15} className="animate-spin" />
                <span>Generating Master Frame & Scene Videos...</span>
              </>
            ) : (
              <>
                <Play size={14} fill="currentColor" />
                <span>Generate Master Frame & Scene Videos</span>
              </>
            )}
          </button>
        </div>
      </div>

      <Handle type="source" position={Position.Right} isConnectable={isConnectable} />
    </div>
  );
});

// ============================================================================
// NODE 2: AI Story Breakdown & Prompt Optimizer Agent (Full Story -> N Scenes)
// ============================================================================
export const NodePlanner = memo(({ data, isConnectable }) => {
  const [selectedSceneIdx, setSelectedSceneIdx] = useState(0);

  const optimizedScenes = data.optimizedScenes || [];
  const safeIdx = selectedSceneIdx < optimizedScenes.length ? selectedSceneIdx : 0;
  const currentScene = optimizedScenes[safeIdx] || (optimizedScenes.length > 0 ? optimizedScenes[0] : null);
  const storyTitle = data.storyTitle || '';

  return (
    <div className={`studio-node planner-node accent-cyan ${data.isActive ? 'active-processing' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />
      <div className="studio-node-header">
        <div className="studio-node-title">
          <BrainCircuit size={16} style={{ color: '#06b6d4' }} />
          <span>2. AI Story Breakdown & Prompt Agent</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <span className="studio-node-badge" style={{ color: '#67e8f9', borderColor: 'rgba(6, 182, 212, 0.35)' }}>
            {data.fluxModel || 'FLUX.1-dev'}
          </span>
          {data.onOptimizePrompts && (
            <button
              type="button"
              className="planner-reopt-all-btn nodrag"
              onClick={(e) => {
                e.stopPropagation();
                data.onOptimizePrompts();
              }}
              disabled={data.isOptimizing}
              title="Re-analyze Full Story & Re-breakdown Scenes"
            >
              {data.isOptimizing ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
              <span>Re-breakdown</span>
            </button>
          )}
        </div>
      </div>

      <div className="studio-node-body">
        {data.liveProgress && data.liveProgress.active && data.liveProgress.node === 'card2' && (
          <div style={{ marginBottom: '0.4rem', padding: '0.4rem 0.6rem', background: 'rgba(6, 182, 212, 0.12)', borderRadius: '6px', border: '1px solid rgba(6, 182, 212, 0.35)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: '#06b6d4', fontWeight: 600, marginBottom: '3px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Loader2 size={11} className="animate-spin" />
                {data.liveProgress.message}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800 }}>{data.liveProgress.percentage}%</span>
            </div>
            <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${data.liveProgress.percentage}%`, background: 'linear-gradient(90deg, #06b6d4, #3b82f6)' }} />
            </div>
          </div>
        )}

        {optimizedScenes.length > 0 ? (
          <>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '0.32rem 0.55rem',
                background: 'rgba(6, 182, 212, 0.1)',
                border: '1px solid rgba(6, 182, 212, 0.3)',
                borderRadius: '6px',
                marginBottom: '0.25rem',
              }}
            >
              <span style={{ fontSize: '0.69rem', fontWeight: 700, color: '#67e8f9' }}>
                🎬 {storyTitle ? storyTitle : 'Story Breakdown'}
              </span>
              <span
                style={{
                  fontSize: '0.65rem',
                  fontWeight: 800,
                  color: '#34d399',
                  background: 'rgba(16, 185, 129, 0.15)',
                  padding: '2px 7px',
                  borderRadius: '4px',
                  border: '1px solid rgba(16, 185, 129, 0.35)',
                }}
              >
                AI Breakdown: {optimizedScenes.length} {optimizedScenes.length === 1 ? 'Scene' : 'Scenes'}
              </span>
            </div>

            <div style={{ display: 'flex', gap: '0.35rem', overflowX: 'auto', paddingBottom: '0.25rem' }}>
              {optimizedScenes.map((sc, idx) => (
                <button
                  key={sc.id || idx}
                  type="button"
                  className={`planner-tab-btn nodrag ${safeIdx === idx ? 'active' : ''}`}
                  onClick={() => setSelectedSceneIdx(idx)}
                  style={{ flexShrink: 0 }}
                >
                  Scene {idx + 1}
                </button>
              ))}
            </div>

            {currentScene && (
              <div className="optimized-scene-display">
                <div className="raw-prompt-preview">
                  <span className="raw-label">
                    <Sparkles size={10} /> Extracted Beat — Scene {safeIdx + 1} of {optimizedScenes.length}:
                  </span>
                  <span>"{currentScene.raw_prompt || '(Extracted story scene)'}"</span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.15rem' }}>
                  <span style={{ fontSize: '0.68rem', color: '#67e8f9', fontWeight: 700, textTransform: 'uppercase' }}>
                    ✨ Scene {safeIdx + 1} {data.fluxModel || 'FLUX'} Master Prompt (Editable):
                  </span>
                  <span style={{ fontSize: '0.64rem', color: '#cbd2de', fontWeight: 600 }}>
                    📐 {data.aspectRatio || '16:9'}
                  </span>
                </div>

                <textarea
                  className="node-textarea nodrag nowheel"
                  style={{ minHeight: '95px', borderLeft: '3px solid #06b6d4', fontSize: '0.72rem' }}
                  value={currentScene.optimized_image_prompt || ''}
                  onChange={(e) =>
                    data.onEditOptimizedPrompt &&
                    data.onEditOptimizedPrompt(safeIdx, e.target.value)
                  }
                  onMouseDown={(e) => e.stopPropagation()}
                />

                <div className="shot-specs-grid">
                  <div><strong>Camera & Lens:</strong> {currentScene.camera}</div>
                  <div><strong>Lighting:</strong> {currentScene.lighting}</div>
                  <div><strong>Environment:</strong> {currentScene.environment}</div>
                </div>

                <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.35rem' }}>
                  <button
                    type="button"
                    className="card-action-btn nodrag"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (data.onRegenerateScene) data.onRegenerateScene(safeIdx);
                    }}
                    disabled={data.isRegeneratingScene === safeIdx || data.isOptimizing}
                  >
                    {data.isRegeneratingScene === safeIdx ? (
                      <>
                        <Loader2 size={12} className="animate-spin" />
                        <span>Re-optimizing...</span>
                      </>
                    ) : (
                      <>
                        <RefreshCw size={12} />
                        <span>Variation Prompt</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    className="card-action-btn nodrag"
                    style={{ background: 'rgba(16, 185, 129, 0.16)', borderColor: '#10b981', color: '#34d399' }}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (data.onGenerateFromOptimized) data.onGenerateFromOptimized();
                    }}
                    disabled={data.isRunning}
                  >
                    <Play size={12} fill="currentColor" />
                    <span>Generate Frame + All {optimizedScenes.length} Scene Videos ➔</span>
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div style={{ fontSize: '0.73rem', background: '#12151c', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center', color: 'var(--text-dim)' }}>
            <BrainCircuit size={26} style={{ margin: '0 auto 0.45rem', opacity: 0.65, color: '#06b6d4' }} />
            <div>Enter your full story in <strong>Card 1</strong> and click <strong>AI Breakdown Story into Scenes</strong>. Gemini AI will automatically determine the scene count and spawn a separate Scene Studio Card for every scene!</div>
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Right} isConnectable={isConnectable} />
    </div>
  );
});

// ============================================================================
// NODE 3: FLUX.1 / FLUX.2 Local GPU Image Generator Studio
// ============================================================================
export const NodeKeyframeGenerator = memo(({ data, isConnectable }) => {
  const [activeFrameIdx, setActiveFrameIdx] = useState(0);

  const frames = data.frames || [];
  const activeFrame = frames[activeFrameIdx] || (frames.length > 0 ? frames[0] : null);

  return (
    <div className={`studio-node keyframe-node accent-emerald ${data.isActive ? 'active-processing' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />
      <div className="studio-node-header">
        <div className="studio-node-title">
          <ImageIcon size={16} style={{ color: '#10b981' }} />
          <span>3. {data.fluxModel || 'Qwen-Image-2.1'} Image Studio</span>
        </div>
        <span
          className="studio-node-badge"
          style={{ color: '#34d399', borderColor: 'rgba(52, 211, 153, 0.35)' }}
          title="Running on Local NVIDIA GeForce RTX 4060 GPU"
        >
          <Cpu size={10} style={{ display: 'inline', marginRight: '3px' }} />
          RTX 4060 Local GPU
        </span>
      </div>

      <div className="studio-node-body">
        {/* Real-Time Local GPU Inference Progress Bar */}
        {data.liveProgress && data.liveProgress.active && data.liveProgress.node === 'card3' && (
          <div style={{ marginBottom: '0.45rem', padding: '0.5rem 0.65rem', background: 'rgba(16, 185, 129, 0.12)', borderRadius: '6px', border: '1px solid rgba(16, 185, 129, 0.35)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#34d399', fontWeight: 600, marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Loader2 size={12} className="animate-spin" />
                {data.liveProgress.message}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800 }}>{data.liveProgress.percentage}%</span>
            </div>
            <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${data.liveProgress.percentage}%`, background: 'linear-gradient(90deg, #10b981, #06b6d4)', transition: 'width 0.3s ease' }} />
            </div>
          </div>
        )}

        {frames.length > 0 ? (
          <>
            {data.folderPath && (
              <div style={{ fontSize: '0.62rem', color: '#94a3b8', background: 'rgba(0,0,0,0.45)', padding: '3px 7px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  📁 Saved: <strong style={{ color: '#ffffff' }}>{data.runId || 'data/keyframes/'}</strong>
                </span>
                <span style={{ fontSize: '0.58rem', color: '#34d399', fontWeight: 700 }}>LOCAL GPU</span>
              </div>
            )}

            {frames.length > 1 && (
              <div style={{ display: 'flex', gap: '0.3rem', overflowX: 'auto', paddingBottom: '0.2rem' }}>
                {frames.map((f, idx) => (
                  <button
                    key={f.scene_number || idx}
                    type="button"
                    className={`planner-tab-btn nodrag ${activeFrameIdx === idx ? 'active' : ''}`}
                    onClick={() => setActiveFrameIdx(idx)}
                  >
                    Scene {idx + 1} Image
                  </button>
                ))}
              </div>
            )}

            {activeFrame && (
              <div className="keyframe-display-card">
                <div className="keyframe-img-wrapper">
                  <img
                    src={activeFrame.imageUrl}
                    alt={`Scene ${activeFrame.scene_number || activeFrameIdx + 1} Generated Image`}
                    className="keyframe-img"
                    onError={(e) => {
                      if (activeFrame.imageUrl && activeFrame.imageUrl.startsWith('/data/') && !e.target.dataset.retried) {
                        e.target.dataset.retried = 'true';
                        e.target.src = `http://localhost:8000${activeFrame.imageUrl}`;
                      }
                    }}
                  />
                  <div className="keyframe-overlay-badge">
                    <span>Scene {activeFrame.scene_number || activeFrameIdx + 1} • {data.fluxModel || 'FLUX.1'} Master Frame</span>
                  </div>
                </div>

                <div className="keyframe-metrics-grid">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-dim)' }}>Inference Engine:</span>
                    <span style={{ color: '#34d399', fontWeight: 600 }}>
                      ⚡ {activeFrame.engine_used || `${data.fluxModel || 'FLUX.1-dev'} (RTX 4060 GPU)`}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-dim)' }}>Camera Lens:</span>
                    <span style={{ color: 'var(--text-main)', fontSize: '0.65rem' }}>
                      {activeFrame.camera || '50mm Anamorphic Prime'}
                    </span>
                  </div>
                  {activeFrame.seed && (
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-dim)' }}>Seed:</span>
                      <span style={{ color: '#cbd2de', fontFamily: 'var(--font-mono)', fontSize: '0.65rem' }}>
                        {activeFrame.seed}
                      </span>
                    </div>
                  )}
                </div>

                <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.25rem' }}>
                  <button
                    type="button"
                    className="card-action-btn nodrag"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (data.onRegenerateFrame) data.onRegenerateFrame(activeFrameIdx);
                    }}
                    disabled={data.isRegeneratingFrame === activeFrameIdx || data.isRunning}
                  >
                    {data.isRegeneratingFrame === activeFrameIdx ? (
                      <>
                        <Loader2 size={13} className="animate-spin" />
                        <span>Synthesizing on GPU...</span>
                      </>
                    ) : (
                      <>
                        <RefreshCw size={13} />
                        <span>Regenerate Frame (GPU)</span>
                      </>
                    )}
                  </button>

                  <a
                    href={activeFrame.imageUrl}
                    download={`flux_scene_${activeFrame.scene_number || 1}.png`}
                    target="_blank"
                    rel="noreferrer"
                    className="card-action-btn nodrag"
                    style={{ width: 'auto', padding: '0.44rem 0.65rem', textDecoration: 'none' }}
                    title="Open / Download Full-Res Image"
                  >
                    <Download size={13} />
                  </a>
                </div>
              </div>
            )}
          </>
        ) : (
          <div style={{ fontSize: '0.73rem', background: '#12151c', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center', color: 'var(--text-dim)' }}>
            <ImageIcon size={26} style={{ margin: '0 auto 0.45rem', opacity: 0.65, color: '#34d399' }} />
            <div>Awaiting optimized prompt from <strong>Card 2</strong>. Synthesizes photorealistic character scenes on your local <strong>NVIDIA GeForce RTX 4060 GPU</strong>.</div>
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Right} isConnectable={isConnectable} />
    </div>
  );
});

// ============================================================================
// Helper: Guarantees Fresh Scene-by-Scene Card 2 Prompt & Card 4 Video/VO Script
// ============================================================================
const CAMERA_VARIATIONS = [
  { cam: 'Medium two-shot at eye level, 50mm anamorphic prime lens, shallow f/1.8 depth of field', motion: 'Slow cinematic dolly push-in' },
  { cam: 'Cinematic medium close-up, 85mm f/1.4 portrait prime lens, creamy bokeh background', motion: 'Subtle handheld breathing motion with slow focus pull' },
  { cam: 'Wide cinematic establishing frame, 35mm ARRI Master Anamorphic lens, rich depth', motion: 'Smooth lateral Steadicam tracking pan' },
  { cam: 'Over-the-shoulder dramatic cinema angle, 50mm Cooke Anamorphic lens, sharp eye focus', motion: 'Slow orbital arc camera movement' },
  { cam: 'Low-angle expressive medium shot, 40mm cinema prime lens, golden lens flare', motion: 'Steady gimbal forward tracking movement' },
  { cam: 'Intimate eye-level cinema frame, 65mm prime lens, soft background separation', motion: 'Gentle crane push-in with subtle parallax' },
];

export function enrichSceneWithVideoAndVO(sc = {}, idx = 0, forceFresh = false, characters = null) {
  const rawText = (sc.raw_prompt || sc.action || sc.text || '').trim();
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const camSpec = pick(CAMERA_VARIATIONS);
  const camMotion = sc.camera_motion || camSpec.motion;
  const camLens = sc.camera || camSpec.cam;

  const charArr = Array.isArray(characters)
    ? characters.map((c) => String(c).toLowerCase())
    : Array.isArray(sc.characters)
      ? sc.characters.map((c) => String(c).toLowerCase())
      : [];
  const wantsMale =
    (charArr.includes('male') && !charArr.includes('female')) ||
    String(sc.voiceover_speaker || '').toLowerCase().includes('arjun') ||
    String(sc.character_description || '').toLowerCase().includes('arjun');

  const speaker = wantsMale ? 'Arjun (Male Lead)' : (sc.voiceover_speaker || 'Maya (Female Lead)');
  const gesture = pick([
    'speaking expressively into the podcast studio microphone with natural lip movements, warm eye contact, and a subtle nod',
    'delivering the line with calm conviction into the studio microphone, tilting the head slightly with a gentle hand gesture',
    'speaking warmly into the condenser microphone, eyebrows lifting naturally with a confident subtle smile',
    'leaning slightly toward the podcast microphone, articulating each word clearly with natural breathing and blinking',
  ]);

  const defaultMotion = `24fps realistic podcast studio video. ${speaker} is ${gesture}. Natural rhythmic lip and jaw articulation, periodic eye blinking, subtle head nods, and steady locked-off studio tripod framing.`;
  const defaultVoSi = sc.voiceover_sinhala || rawText || 'Spoken script segment';
  const defaultVoEn = sc.voiceover_english || rawText || 'Spoken script segment';
  const defaultSfx = sc.sfx_cue || 'Clean studio room tone, warm vocal presence';

  let optImgPrompt = sc.optimized_image_prompt || '';
  if (!optImgPrompt) {
    const backdrop = pick([
      'deep maroon velvet acoustic studio curtains with subtle warm rim light',
      'warm dark walnut wood-slat acoustic studio panels with a soft amber LED accent strip',
      'emerald-green soundproof plush studio panels with moody cinematic warm tungsten key lighting',
      'charcoal-grey modern podcast booth walls with warm golden practical sconces in the background',
    ]);
    const mic = pick([
      'a sleek rose-gold and copper broadcast condenser podcast microphone on a studio boom arm',
      'a professional matte-black dynamic broadcast podcast microphone on a low-profile studio boom arm',
      'a warm champagne-gold broadcast podcast microphone on a studio desk arm',
    ]);
    const mug = pick([
      'a matte black ceramic coffee mug on the foreground studio table',
      'a dark stoneware artisan coffee mug on the corner of the wooden podcast desk',
      'a deep burgundy ceramic coffee cup sitting in soft foreground bokeh on the studio desk',
    ]);
    const subjDesc = wantsMale
      ? 'Arjun, a handsome young South Asian man in his mid-20s (solo male podcast host, 1man, no woman in frame) with thick wavy black hair, a neatly trimmed full beard, wearing a fitted matte-black crew-neck t-shirt and a classic black wristwatch'
      : 'Maya, a gorgeous young South Asian woman in her early 20s (solo female podcast host, 1girl, no man in frame) with long dark wavy hair, wearing a stylish dark studio top';
    optImgPrompt = `Photorealistic podcast studio talking-head master frame of ${subjDesc}, seated comfortably in a professional podcast studio speaking into ${mic}. ${camLens}. Behind the speaker is ${backdrop}. In the lower foreground rests ${mug}. Warm broadcast studio lighting, 8k RAW cinema photography.`;
  }

  return {
    ...sc,
    characters: wantsMale ? ['male'] : ['female'],
    camera: camLens,
    camera_motion: camMotion,
    optimized_image_prompt: optImgPrompt,
    video_motion_prompt: (!forceFresh && sc.video_motion_prompt && sc.video_motion_prompt.trim()) ? sc.video_motion_prompt : defaultMotion,
    voiceover_speaker: speaker,
    voiceover_sinhala: (!forceFresh && sc.voiceover_sinhala && sc.voiceover_sinhala.trim()) ? sc.voiceover_sinhala : defaultVoSi,
    voiceover_english: (!forceFresh && sc.voiceover_english && sc.voiceover_english.trim()) ? sc.voiceover_english : defaultVoEn,
    sfx_cue: (!forceFresh && sc.sfx_cue && sc.sfx_cue.trim()) ? sc.sfx_cue : defaultSfx,
  };
}

// ============================================================================
// NODE 4: Short Film Video Motion & Voice-Over Script Studio (Parallel to Card 3)
// ============================================================================
export const NodeVideoScriptStudio = memo(({ data, isConnectable }) => {
  const [activeSceneIdx, setActiveSceneIdx] = useState(0);
  const [copiedField, setCopiedField] = useState(null);

  const rawScenes = data.optimizedScenes || [];
  const scenes = rawScenes.map((sc, idx) => enrichSceneWithVideoAndVO(sc, idx));
  const totalScenes = scenes.length;
  const safeIdx = activeSceneIdx < totalScenes ? activeSceneIdx : 0;
  const currentScene = scenes[safeIdx] || {};

  const handleCopy = (text, key) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedField(key);
    setTimeout(() => setCopiedField(null), 1600);
  };

  const speakVoiceOver = (text) => {
    if (!text || typeof window === 'undefined' || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.94;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className={`studio-node video-script-node accent-amber ${data.isActive ? 'active-processing' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />
      <div className="studio-node-header">
        <div className="studio-node-title">
          <Wand2 size={16} style={{ color: '#f59e0b' }} />
          <span>4. Video Motion & Voice-Over Script</span>
        </div>
        <span
          className="studio-node-badge"
          style={{ color: '#fcd34d', borderColor: 'rgba(245, 158, 11, 0.4)' }}
        >
          Parallel to Card 3
        </span>
      </div>

      <div className="studio-node-body">
        {totalScenes > 0 ? (
          <>
            {/* Scene-by-Scene Tabs (Parallel to Card 3) */}
            {totalScenes > 1 && (
              <div style={{ display: 'flex', gap: '0.35rem', overflowX: 'auto', paddingBottom: '0.15rem' }}>
                {scenes.map((sc, idx) => (
                  <button
                    key={sc.id || idx}
                    type="button"
                    className={`planner-tab-btn nodrag ${safeIdx === idx ? 'active' : ''}`}
                    onClick={() => setActiveSceneIdx(idx)}
                  >
                    Scene {idx + 1} Script & VO
                  </button>
                ))}
              </div>
            )}

            <div className="optimized-scene-display" style={{ gap: '0.45rem' }}>
              {/* Camera Movement Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#07090d', padding: '0.35rem 0.55rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)', fontSize: '0.65rem' }}>
                <span style={{ color: 'var(--text-dim)' }}>
                  🎬 Camera Motion: <strong style={{ color: '#fcd34d' }}>{currentScene.camera_motion || 'Slow dolly push-in'}</strong>
                </span>
                <span style={{ color: '#34d399', fontWeight: 700 }}>⏱️ {currentScene.duration || '5s'}</span>
              </div>

              {/* 1. Video Motion & Action Script (What happens in the video) */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
                  <span style={{ fontSize: '0.66rem', color: '#fcd34d', fontWeight: 700, textTransform: 'uppercase' }}>
                    🎥 1. Video Motion Script (What Happens in Scene {safeIdx + 1}):
                  </span>
                  <button
                    type="button"
                    className="planner-reopt-all-btn nodrag"
                    onClick={() => handleCopy(currentScene.video_motion_prompt || '', `motion_${safeIdx}`)}
                  >
                    {copiedField === `motion_${safeIdx}` ? <Check size={10} /> : <Copy size={10} />}
                    <span>{copiedField === `motion_${safeIdx}` ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <textarea
                  className="node-textarea nodrag nowheel"
                  style={{ minHeight: '72px', borderLeft: '3px solid #f59e0b', fontSize: '0.7rem' }}
                  value={currentScene.video_motion_prompt || ''}
                  onChange={(e) =>
                    data.onEditVideoScriptField &&
                    data.onEditVideoScriptField(safeIdx, 'video_motion_prompt', e.target.value)
                  }
                  onMouseDown={(e) => e.stopPropagation()}
                  placeholder="Describes exact character actions, facial expressions, environment movement, and camera motion..."
                />
              </div>

              {/* 2. Short Film Voice-Over (VO) & Dialogue Script */}
              <div style={{ background: '#080a0f', padding: '0.45rem 0.55rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.09)', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.66rem', color: '#67e8f9', fontWeight: 700, textTransform: 'uppercase' }}>
                    🎙️ 2. Short Film Voice-Over (VO) / Dialogue:
                  </span>
                  <span style={{ fontSize: '0.61rem', color: '#d8b4fe', fontWeight: 600 }}>
                    🗣️ {currentScene.voiceover_speaker || 'Arjun (Male Lead)'}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.63rem', color: 'var(--text-dim)' }}>
                    🇱🇰 <strong>Sinhala / Singlish VO:</strong>
                  </span>
                  <button
                    type="button"
                    className="planner-reopt-all-btn nodrag"
                    onClick={() => handleCopy(currentScene.voiceover_sinhala || '', `vosi_${safeIdx}`)}
                  >
                    {copiedField === `vosi_${safeIdx}` ? <Check size={10} /> : <Copy size={10} />}
                    <span>{copiedField === `vosi_${safeIdx}` ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <textarea
                  className="node-textarea nodrag nowheel"
                  style={{ minHeight: '48px', borderLeft: '3px solid #a855f7', fontSize: '0.69rem', padding: '0.4rem 0.55rem' }}
                  value={currentScene.voiceover_sinhala || ''}
                  onChange={(e) =>
                    data.onEditVideoScriptField &&
                    data.onEditVideoScriptField(safeIdx, 'voiceover_sinhala', e.target.value)
                  }
                  onMouseDown={(e) => e.stopPropagation()}
                />

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.63rem', color: 'var(--text-dim)' }}>
                    🌐 <strong>English VO / Subtitle:</strong>
                  </span>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <button
                      type="button"
                      className="planner-reopt-all-btn nodrag"
                      onClick={() => handleCopy(currentScene.voiceover_english || '', `voen_${safeIdx}`)}
                    >
                      {copiedField === `voen_${safeIdx}` ? <Check size={10} /> : <Copy size={10} />}
                      <span>{copiedField === `voen_${safeIdx}` ? 'Copied' : 'Copy'}</span>
                    </button>
                    <button
                      type="button"
                      className="planner-reopt-all-btn nodrag"
                      onClick={() => speakVoiceOver(currentScene.voiceover_english || currentScene.raw_prompt)}
                      title="Listen to Voice-Over Audio"
                    >
                      <Volume2 size={11} style={{ color: '#67e8f9' }} />
                      <span>Play VO Audio</span>
                    </button>
                  </div>
                </div>
                <textarea
                  className="node-textarea nodrag nowheel"
                  style={{ minHeight: '44px', borderLeft: '3px solid #06b6d4', fontSize: '0.69rem', padding: '0.4rem 0.55rem' }}
                  value={currentScene.voiceover_english || ''}
                  onChange={(e) =>
                    data.onEditVideoScriptField &&
                    data.onEditVideoScriptField(safeIdx, 'voiceover_english', e.target.value)
                  }
                  onMouseDown={(e) => e.stopPropagation()}
                />

                <div style={{ fontSize: '0.63rem', color: '#94a3b8', display: 'flex', justifyContent: 'space-between', paddingTop: '2px' }}>
                  <span>🎵 <strong>SFX:</strong> {currentScene.sfx_cue || 'Natural cinema ambience'}</span>
                </div>
              </div>

              {/* Action Button in Card 4 */}
              <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.2rem' }}>
                <button
                  type="button"
                  className="card-action-btn nodrag"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (data.onRegenerateScene) data.onRegenerateScene(safeIdx);
                  }}
                  disabled={data.isRegeneratingScene === safeIdx}
                >
                  <RefreshCw size={12} />
                  <span>Regenerate Scene {safeIdx + 1} Script & Voice-Over</span>
                </button>
              </div>
            </div>
          </>
        ) : (
          <div style={{ fontSize: '0.73rem', background: '#12151c', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center', color: 'var(--text-dim)' }}>
            <Wand2 size={26} style={{ margin: '0 auto 0.45rem', opacity: 0.65, color: '#f59e0b' }} />
            <div>
              <strong>Parallel to Card 3:</strong> Generates Scene-by-Scene <strong>Video Motion Script</strong> (what happens in each shot), <strong>Short Film Voice-Overs (Sinhala & English)</strong>, and <strong>SFX Cues</strong>!
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

// ============================================================================
// NODE 5: Wan 2.2 Video Generator Studio (Combines Card 3 Frame + Card 4 Script)
// ============================================================================
export const NodeWanVideoGenerator = memo(({ data, isConnectable }) => {
  const [activeSceneIdx, setActiveSceneIdx] = useState(0);

  const scenes = data.optimizedScenes || [];
  const frames = data.frames || [];
  const videos = data.videos || [];
  const wanModel = data.wanModel || 'Wan2.2-TI2V-5B';

  const totalScenes = Math.max(scenes.length, frames.length, videos.length);
  const safeIdx = activeSceneIdx < totalScenes ? activeSceneIdx : 0;

  const currentScene = scenes[safeIdx] || {};
  const currentFrame = frames[safeIdx] || null;
  const currentVideo = videos[safeIdx] || null;

  const bothReady = Boolean(currentFrame && currentFrame.imageUrl && currentScene && currentScene.video_motion_prompt);

  const speakVoiceOver = (text) => {
    if (!text || typeof window === 'undefined' || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.94;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className={`studio-node wan-video-node accent-rose ${data.isActive ? 'active-processing' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />
      <div className="studio-node-header">
        <div className="studio-node-title">
          <Film size={16} style={{ color: '#f43f5e' }} />
          <span>5. Wan 2.2 Video Generator Studio</span>
        </div>
        <span
          className="studio-node-badge"
          style={{ color: '#fda4af', borderColor: 'rgba(244, 63, 94, 0.4)' }}
        >
          Card 3 + Card 4 ➔ MP4
        </span>
      </div>

      <div className="studio-node-body">
        {/* Wan 2.2 Model Selector */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '0.4rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600 }}>Wan 2.2 Engine:</span>
          <div className="selector-chip-group">
            {['Wan2.2-TI2V-5B', 'Wan2.2-I2V-A14B', 'Wan2.1-I2V-14B'].map((wm) => (
              <button
                key={wm}
                type="button"
                className={`selector-chip nodrag ${wanModel === wm ? 'active' : ''}`}
                onClick={() => data.onSelectWanModel && data.onSelectWanModel(wm)}
              >
                {wm}
              </button>
            ))}
          </div>
        </div>

        {/* Live Progress Bar for Card 5 Video Synthesis */}
        {data.liveProgress && data.liveProgress.active && data.liveProgress.node === 'card4' && (
          <div style={{ padding: '0.45rem 0.6rem', background: 'rgba(244, 63, 94, 0.14)', borderRadius: '6px', border: '1px solid rgba(244, 63, 94, 0.4)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: '#fda4af', fontWeight: 600, marginBottom: '3px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Loader2 size={11} className="animate-spin" />
                {data.liveProgress.message}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800 }}>{data.liveProgress.percentage}%</span>
            </div>
            <div style={{ height: '5px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${data.liveProgress.percentage}%`, background: 'linear-gradient(90deg, #f43f5e, #f59e0b)', transition: 'width 0.25s ease' }} />
            </div>
          </div>
        )}

        {totalScenes > 0 ? (
          <>
            {/* Scene-by-Scene Video Tabs */}
            {totalScenes > 1 && (
              <div style={{ display: 'flex', gap: '0.35rem', overflowX: 'auto', paddingBottom: '0.15rem' }}>
                {Array.from({ length: totalScenes }).map((_, idx) => (
                  <button
                    key={idx}
                    type="button"
                    className={`planner-tab-btn nodrag ${safeIdx === idx ? 'active' : ''}`}
                    onClick={() => setActiveSceneIdx(idx)}
                  >
                    Scene {idx + 1} Video
                  </button>
                ))}
              </div>
            )}

            <div className="optimized-scene-display" style={{ gap: '0.45rem' }}>
              {/* Dual Inputs Status Box: Card 3 Frame + Card 4 Script */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', background: '#07090d', padding: '0.45rem 0.55rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)' }}>
                {currentFrame && currentFrame.imageUrl ? (
                  <img
                    src={currentFrame.imageUrl}
                    alt={`Scene ${safeIdx + 1} Keyframe`}
                    style={{ width: '64px', height: '42px', objectFit: 'cover', borderRadius: '5px', border: '1px solid rgba(52, 211, 153, 0.6)' }}
                  />
                ) : (
                  <div style={{ width: '64px', height: '42px', borderRadius: '5px', background: '#11151f', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.58rem', color: 'var(--text-dim)', border: '1px dashed rgba(255,255,255,0.2)' }}>
                    Card 3
                  </div>
                )}
                <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <div style={{ fontSize: '0.65rem', display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#cbd2de', fontWeight: 600 }}>🖼️ Card 3 Frame Image:</span>
                    <span style={{ color: currentFrame ? '#34d399' : '#f59e0b', fontWeight: 700 }}>
                      {currentFrame ? '✓ Ready' : '⏳ Pending'}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.65rem', display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: '#cbd2de', fontWeight: 600 }}>📜 Card 4 Motion & VO Script:</span>
                    <span style={{ color: currentScene.video_motion_prompt ? '#fcd34d' : '#94a3b8', fontWeight: 700 }}>
                      {currentScene.video_motion_prompt ? '✓ Ready' : '⏳ Pending'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Generated Video Player */}
              {currentVideo && currentVideo.videoUrl ? (
                <div style={{ background: '#05070b', padding: '0.5rem', borderRadius: '7px', border: '1px solid rgba(244, 63, 94, 0.35)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem', fontSize: '0.66rem' }}>
                    <span style={{ color: '#fda4af', fontWeight: 700 }}>
                      🎬 Scene {safeIdx + 1} Master Video ({currentVideo.wan_model || wanModel})
                    </span>
                    <a
                      href={currentVideo.videoUrl}
                      download={`scene_${safeIdx + 1}_wan22.mp4`}
                      target="_blank"
                      rel="noreferrer"
                      style={{ color: '#34d399', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px', fontWeight: 600 }}
                    >
                      <Download size={11} /> Download MP4
                    </a>
                  </div>
                  <video
                    key={currentVideo.videoUrl}
                    src={currentVideo.videoUrl}
                    controls
                    loop
                    playsInline
                    onPlay={() => {
                      if (currentScene.voiceover_english) {
                        speakVoiceOver(currentScene.voiceover_english);
                      }
                    }}
                    style={{ width: '100%', maxHeight: '215px', borderRadius: '6px', background: '#000', objectFit: 'cover' }}
                  />
                  <div style={{ marginTop: '0.35rem', fontSize: '0.64rem', color: '#cbd2de', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      🎙️ <strong>{currentScene.voiceover_speaker || 'VO'}:</strong> "{currentScene.voiceover_english || ''}"
                    </span>
                    <button
                      type="button"
                      className="planner-reopt-all-btn nodrag"
                      onClick={() => speakVoiceOver(currentScene.voiceover_english)}
                    >
                      <Volume2 size={10} /> VO
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: '0.7rem', background: '#0c0f16', padding: '0.85rem', borderRadius: '7px', border: '1px dashed rgba(244, 63, 94, 0.35)', textAlign: 'center', color: 'var(--text-dim)' }}>
                  {bothReady
                    ? '✓ Both Card 3 (Frame Image) & Card 4 (Motion/VO Script) are ready! Click Generate below to render the Wan 2.2 Video.'
                    : 'Waiting for Card 3 (Master Frame Image) and Card 4 (Motion & Voice-Over Script) to complete...'}
                </div>
              )}

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.2rem' }}>
                <button
                  type="button"
                  className="card-action-btn nodrag"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (data.onRegenerateSingleVideo) data.onRegenerateSingleVideo(safeIdx);
                  }}
                  disabled={!bothReady || data.isGeneratingVideos || data.isRegeneratingVideo === safeIdx}
                >
                  {data.isRegeneratingVideo === safeIdx ? (
                    <>
                      <Loader2 size={12} className="animate-spin" />
                      <span>Rendering S{safeIdx + 1}...</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw size={12} />
                      <span>Render Scene {safeIdx + 1} Video</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="card-action-btn nodrag"
                  style={{ background: 'rgba(244, 63, 94, 0.2)', borderColor: '#f43f5e', color: '#fda4af' }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (data.onGenerateAllVideos) data.onGenerateAllVideos();
                  }}
                  disabled={!bothReady || data.isGeneratingVideos}
                >
                  {data.isGeneratingVideos ? (
                    <>
                      <Loader2 size={12} className="animate-spin" />
                      <span>Synthesizing Videos...</span>
                    </>
                  ) : (
                    <>
                      <Play size={12} fill="currentColor" />
                      <span>Generate All Videos ({wanModel})</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </>
        ) : (
          <div style={{ fontSize: '0.73rem', background: '#12151c', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center', color: 'var(--text-dim)' }}>
            <Film size={26} style={{ margin: '0 auto 0.45rem', opacity: 0.65, color: '#f43f5e' }} />
            <div>
              Awaiting <strong>Card 3 (Generated Frame Image)</strong> + <strong>Card 4 (Motion & Voice-Over Script)</strong>. Once both parallel branches finish, <strong>Card 5</strong> synthesizes the final <strong>Wan 2.2 .MP4 Video</strong>!
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

// ============================================================================
// UNIFIED DYNAMIC SCENE STUDIO CARD (Combines Frame Image + Motion & VO Script per Scene)
// ============================================================================
export const NodeSceneStudio = memo(({ data, isConnectable }) => {
  const [copiedField, setCopiedField] = useState(null);

  const sceneIdx = data.sceneIndex ?? 0;
  const sceneNum = data.sceneNumber ?? sceneIdx + 1;
  const fluxModel = data.fluxModel || 'Qwen-Image-2.1';
  const frame = data.frame || null;
  const rawScript = data.sceneScript || null;
  const sceneScript = rawScript ? enrichSceneWithVideoAndVO(rawScript, sceneIdx, false) : null;

  const isThisSceneRendering =
    data.liveProgress &&
    data.liveProgress.active &&
    data.liveProgress.node === 'card3' &&
    (!data.liveProgress.current_scene || data.liveProgress.current_scene === sceneNum);

  const handleCopy = (text, key) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedField(key);
    setTimeout(() => setCopiedField(null), 1600);
  };

  const speakVoiceOver = (text) => {
    if (!text || typeof window === 'undefined' || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.94;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  const accentColors = ['#10b981', '#f59e0b', '#38bdf8'];
  const badgeColor = accentColors[sceneIdx % accentColors.length];

  return (
    <div
      className={`studio-node keyframe-node accent-emerald ${data.isActive || isThisSceneRendering ? 'active-processing' : ''}`}
      style={{ width: '435px', borderColor: data.isActive || isThisSceneRendering ? badgeColor : undefined }}
    >
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />

      <div className="studio-node-header">
        <div className="studio-node-title">
          <Film size={16} style={{ color: badgeColor }} />
          <span>3.{sceneNum} Scene {sceneNum} Studio — Frame & Script</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <span
            className="studio-node-badge"
            style={{ color: badgeColor, borderColor: `${badgeColor}55` }}
            title="Combined Scene Master Frame Image + Video Motion & Voice-Over Script"
          >
            <Cpu size={10} style={{ display: 'inline', marginRight: '3px' }} />
            Scene {sceneNum} • {fluxModel}
          </span>
        </div>
      </div>

      <div className="studio-node-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        {/* Live Local GPU Progress Bar for this Scene */}
        {isThisSceneRendering && (
          <div style={{ padding: '0.5rem 0.65rem', background: 'rgba(16, 185, 129, 0.14)', borderRadius: '6px', border: '1px solid rgba(16, 185, 129, 0.4)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#34d399', fontWeight: 600, marginBottom: '4px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Loader2 size={12} className="animate-spin" />
                {data.liveProgress.message}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800 }}>{data.liveProgress.percentage}%</span>
            </div>
            <div style={{ height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${data.liveProgress.percentage}%`, background: 'linear-gradient(90deg, #10b981, #06b6d4)', transition: 'width 0.3s ease' }} />
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* PART A: SCENE MASTER FRAME IMAGE (TOP SECTION)                   */}
        {/* ================================================================ */}
        <div style={{ background: '#0b0e15', padding: '0.55rem', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.28)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#34d399', display: 'flex', alignItems: 'center', gap: '5px', textTransform: 'uppercase' }}>
              <ImageIcon size={13} />
              Scene {sceneNum} Master Frame Image
            </span>
            {data.runId && (
              <span style={{ fontSize: '0.6rem', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
                {data.runId}
              </span>
            )}
          </div>

          {frame && frame.imageUrl ? (
            <div className="keyframe-display-card">
              <div className="keyframe-img-wrapper">
                <img
                  src={frame.imageUrl}
                  alt={`Scene ${sceneNum} Generated Master Frame`}
                  className="keyframe-img"
                  onError={(e) => {
                    if (frame.imageUrl && frame.imageUrl.startsWith('/data/') && !e.target.dataset.retried) {
                      e.target.dataset.retried = 'true';
                      e.target.src = `http://localhost:8000${frame.imageUrl}`;
                    }
                  }}
                />
                <div className="keyframe-overlay-badge">
                  <span>Scene {sceneNum} • {fluxModel} Master Frame</span>
                </div>
              </div>

              <div className="keyframe-metrics-grid" style={{ marginTop: '0.35rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-dim)' }}>Engine:</span>
                  <span style={{ color: '#34d399', fontWeight: 600 }}>
                    ⚡ {frame.engine_used || `${fluxModel} (RTX 4060 GPU)`}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-dim)' }}>Camera Lens:</span>
                  <span style={{ color: 'var(--text-main)', fontSize: '0.64rem' }}>
                    {frame.camera || sceneScript?.camera || '50mm Anamorphic Prime'}
                  </span>
                </div>
                {frame.seed && (
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span style={{ color: 'var(--text-dim)' }}>Seed:</span>
                    <span style={{ color: '#cbd2de', fontFamily: 'var(--font-mono)', fontSize: '0.64rem' }}>
                      {frame.seed}
                    </span>
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.35rem' }}>
                <button
                  type="button"
                  className="card-action-btn nodrag"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (data.onRegenerateFrame) data.onRegenerateFrame(sceneIdx);
                  }}
                  disabled={data.isRegeneratingFrame === sceneIdx || data.isRunning}
                >
                  {data.isRegeneratingFrame === sceneIdx ? (
                    <>
                      <Loader2 size={13} className="animate-spin" />
                      <span>Synthesizing Scene {sceneNum} on GPU...</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw size={13} />
                      <span>Regenerate Scene {sceneNum} Frame</span>
                    </>
                  )}
                </button>

                <a
                  href={frame.imageUrl}
                  download={`scene_${sceneNum}_master.png`}
                  target="_blank"
                  rel="noreferrer"
                  className="card-action-btn nodrag"
                  style={{ width: 'auto', padding: '0.44rem 0.65rem', textDecoration: 'none' }}
                  title="Download Full-Res Scene Frame"
                >
                  <Download size={13} />
                </a>
              </div>
            </div>
          ) : (
            <div style={{ fontSize: '0.7rem', background: '#11151e', padding: '0.85rem', borderRadius: '7px', border: '1px dashed rgba(16, 185, 129, 0.35)', textAlign: 'center', color: 'var(--text-dim)' }}>
              {data.isRunning || isThisSceneRendering ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem', color: '#34d399' }}>
                  <Loader2 size={20} className="animate-spin" />
                  <span>Synthesizing Scene {sceneNum} Master Frame on Local RTX 4060 GPU...</span>
                </div>
              ) : (
                <div>
                  Scene {sceneNum} Frame Image will render here when you click <strong>Optimize & Generate Image</strong> (or <strong>Generate Scene Frame(s)</strong> in Card 2).
                </div>
              )}
            </div>
          )}
        </div>

        {/* ================================================================ */}
        {/* PART B: SCENE VIDEO MOTION & VOICE-OVER SCRIPT (BOTTOM SECTION)  */}
        {/* ================================================================ */}
        <div style={{ background: '#0b0e15', padding: '0.55rem', borderRadius: '8px', border: '1px solid rgba(245, 158, 11, 0.3)', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#fcd34d', display: 'flex', alignItems: 'center', gap: '5px', textTransform: 'uppercase' }}>
              <Wand2 size={13} />
              Scene {sceneNum} Video Motion & Voice-Over Script
            </span>
            <span style={{ fontSize: '0.62rem', color: '#34d399', fontWeight: 700 }}>
              ⏱️ {sceneScript?.duration || '5s'}
            </span>
          </div>

          {sceneScript ? (
            <>
              {/* Camera Movement Strip */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#07090d', padding: '0.32rem 0.5rem', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.08)', fontSize: '0.64rem' }}>
                <span style={{ color: 'var(--text-dim)' }}>
                  🎬 Camera Motion: <strong style={{ color: '#fcd34d' }}>{sceneScript.camera_motion || 'Slow cinematic dolly push-in'}</strong>
                </span>
              </div>

              {/* 1. Video Motion Script */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.2rem' }}>
                  <span style={{ fontSize: '0.64rem', color: '#fcd34d', fontWeight: 700 }}>
                    🎥 1. Wan 2.2 Video Motion Script (Scene {sceneNum}):
                  </span>
                  <button
                    type="button"
                    className="planner-reopt-all-btn nodrag"
                    onClick={() => handleCopy(sceneScript.video_motion_prompt || '', `motion_${sceneIdx}`)}
                  >
                    {copiedField === `motion_${sceneIdx}` ? <Check size={10} /> : <Copy size={10} />}
                    <span>{copiedField === `motion_${sceneIdx}` ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <textarea
                  className="node-textarea nodrag nowheel"
                  style={{ minHeight: '64px', borderLeft: '3px solid #f59e0b', fontSize: '0.68rem' }}
                  value={sceneScript.video_motion_prompt || ''}
                  onChange={(e) =>
                    data.onEditVideoScriptField &&
                    data.onEditVideoScriptField(sceneIdx, 'video_motion_prompt', e.target.value)
                  }
                  onMouseDown={(e) => e.stopPropagation()}
                />
              </div>

              {/* 2. Sinhala & English Voice-Over Script */}
              <div style={{ background: '#080a0f', padding: '0.42rem 0.5rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.08)', display: 'flex', flexDirection: 'column', gap: '0.32rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.64rem', color: '#67e8f9', fontWeight: 700 }}>
                    🎙️ 2. Voice-Over (VO) / Dialogue:
                  </span>
                  <span style={{ fontSize: '0.6rem', color: '#d8b4fe', fontWeight: 600 }}>
                    🗣️ {sceneScript.voiceover_speaker || 'Arjun (Male Lead)'}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.62rem', color: 'var(--text-dim)' }}>
                    🇱🇰 <strong>Sinhala / Singlish VO:</strong>
                  </span>
                  <button
                    type="button"
                    className="planner-reopt-all-btn nodrag"
                    onClick={() => handleCopy(sceneScript.voiceover_sinhala || '', `vosi_${sceneIdx}`)}
                  >
                    {copiedField === `vosi_${sceneIdx}` ? <Check size={10} /> : <Copy size={10} />}
                    <span>{copiedField === `vosi_${sceneIdx}` ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <textarea
                  className="node-textarea nodrag nowheel"
                  style={{ minHeight: '44px', borderLeft: '3px solid #a855f7', fontSize: '0.68rem', padding: '0.35rem 0.5rem' }}
                  value={sceneScript.voiceover_sinhala || ''}
                  onChange={(e) =>
                    data.onEditVideoScriptField &&
                    data.onEditVideoScriptField(sceneIdx, 'voiceover_sinhala', e.target.value)
                  }
                  onMouseDown={(e) => e.stopPropagation()}
                />

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.62rem', color: 'var(--text-dim)' }}>
                    🌐 <strong>English VO / Subtitle:</strong>
                  </span>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <button
                      type="button"
                      className="planner-reopt-all-btn nodrag"
                      onClick={() => handleCopy(sceneScript.voiceover_english || '', `voen_${sceneIdx}`)}
                    >
                      {copiedField === `voen_${sceneIdx}` ? <Check size={10} /> : <Copy size={10} />}
                      <span>{copiedField === `voen_${sceneIdx}` ? 'Copied' : 'Copy'}</span>
                    </button>
                    <button
                      type="button"
                      className="planner-reopt-all-btn nodrag"
                      onClick={() => speakVoiceOver(sceneScript.voiceover_english || sceneScript.raw_prompt)}
                      title="Listen to Voice-Over Audio"
                    >
                      <Volume2 size={11} style={{ color: '#67e8f9' }} />
                      <span>Play VO</span>
                    </button>
                  </div>
                </div>
                <textarea
                  className="node-textarea nodrag nowheel"
                  style={{ minHeight: '42px', borderLeft: '3px solid #06b6d4', fontSize: '0.68rem', padding: '0.35rem 0.5rem' }}
                  value={sceneScript.voiceover_english || ''}
                  onChange={(e) =>
                    data.onEditVideoScriptField &&
                    data.onEditVideoScriptField(sceneIdx, 'voiceover_english', e.target.value)
                  }
                  onMouseDown={(e) => e.stopPropagation()}
                />

                <div style={{ fontSize: '0.61rem', color: '#94a3b8', paddingTop: '2px' }}>
                  🎵 <strong>SFX:</strong> {sceneScript.sfx_cue || 'Natural cinema ambience'}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button
                  type="button"
                  className="card-action-btn nodrag"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (data.onRegenerateScene) data.onRegenerateScene(sceneIdx);
                  }}
                  disabled={data.isRegeneratingScene === sceneIdx}
                >
                  {data.isRegeneratingScene === sceneIdx ? (
                    <>
                      <Loader2 size={12} className="animate-spin" />
                      <span>Updating Script...</span>
                    </>
                  ) : (
                    <>
                      <RefreshCw size={12} />
                      <span>Regenerate S{sceneNum} Script</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  className="card-action-btn nodrag"
                  style={{
                    background: 'rgba(244, 63, 94, 0.2)',
                    borderColor: '#f43f5e',
                    color: '#fda4af',
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (data.onGenerateSceneVideo) data.onGenerateSceneVideo(sceneIdx);
                  }}
                  disabled={!frame || data.isGeneratingVideos || data.isRegeneratingVideo === sceneIdx}
                  title={`Synthesize Scene ${sceneNum} Video in Card 4.${sceneNum} using ${data.wanModel || 'LTX-Video-2B'}`}
                >
                  {data.isRegeneratingVideo === sceneIdx ? (
                    <>
                      <Loader2 size={12} className="animate-spin" />
                      <span>Rendering 4.{sceneNum}...</span>
                    </>
                  ) : (
                    <>
                      <Film size={12} />
                      <span>Generate 4.{sceneNum} Video ➔</span>
                    </>
                  )}
                </button>
              </div>
            </>
          ) : (
            <div style={{ fontSize: '0.7rem', background: '#11151e', padding: '0.75rem', borderRadius: '7px', border: '1px dashed rgba(245, 158, 11, 0.3)', textAlign: 'center', color: 'var(--text-dim)' }}>
              Scene {sceneNum} Video Motion & Voice-Over Script will appear here once Card 2 optimizes your scenario.
            </div>
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Right} isConnectable={isConnectable} />
    </div>
  );
});

// ============================================================================
// DYNAMIC SCENE VIDEO STUDIO CARD (4.1, 4.2, ..., 4.N — Connected 3.K -> 4.K)
// ============================================================================
export const NodeSceneVideo = memo(({ data, isConnectable }) => {
  const sceneIdx = data.sceneIndex ?? 0;
  const sceneNum = data.sceneNumber || sceneIdx + 1;
  const wanModel = data.wanModel || 'Wan2.2-I2V-A14B';
  const frame = data.frame || null;
  const sceneScript = data.sceneScript ? enrichSceneWithVideoAndVO(data.sceneScript, sceneIdx, false) : null;
  const videoClip = data.videoClip || null;

  const isThisVideoRendering =
    (data.liveProgress &&
      data.liveProgress.active &&
      data.liveProgress.node === 'card4' &&
      (!data.liveProgress.current_scene || data.liveProgress.current_scene === sceneNum)) ||
    data.isRegeneratingVideo === sceneIdx;

  const bothReady = Boolean(frame && frame.imageUrl && sceneScript);

  const speakVoiceOver = (text) => {
    if (!text || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 0.96;
    utter.pitch = 1.0;
    window.speechSynthesis.speak(utter);
  };

  const accentColors = ['#f43f5e', '#ec4899', '#f59e0b', '#38bdf8', '#a855f7', '#10b981'];
  const badgeColor = accentColors[sceneIdx % accentColors.length];

  return (
    <div
      className={`studio-node video-generator-node accent-rose ${data.isActive || isThisVideoRendering ? 'active-processing' : ''}`}
      style={{ width: '435px' }}
    >
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />

      <div className="studio-node-header">
        <div className="studio-node-title">
          <Film size={16} style={{ color: badgeColor }} />
          <span>4.{sceneNum} Scene {sceneNum} Video Studio</span>
        </div>
        <span
          className="studio-node-badge"
          style={{
            color: badgeColor,
            borderColor: `${badgeColor}66`,
            background: `${badgeColor}18`,
          }}
        >
          3.{sceneNum} ➔ 4.{sceneNum} • {wanModel}
        </span>
      </div>

      <div className="studio-node-body" style={{ gap: '0.5rem' }}>
        {/* Video Model Selector Strip (LTX & Wan 2.2) */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingBottom: '0.35rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <span style={{ fontSize: '0.67rem', color: 'var(--text-dim)', fontWeight: 600 }}>Video Model:</span>
          <div className="selector-chip-group">
            {['Wan2.2-I2V-A14B', 'Wan2.2-T2V-A14B', 'LTX-Video-13B', 'LTX-Video-2B'].map((wm) => (
              <button
                key={wm}
                type="button"
                className={`selector-chip nodrag ${wanModel === wm ? 'active' : ''}`}
                style={{ fontSize: '0.59rem', padding: '2px 6px' }}
                onClick={() => data.onSelectWanModel && data.onSelectWanModel(wm)}
              >
                {wm.replace('Wan2.2-', 'Wan ').replace('-Video-', ' ')}
              </button>
            ))}
          </div>
        </div>

        {/* Source Input Summary from Card 3.K (Master Frame + Scene K Script) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', background: '#07090d', padding: '0.42rem 0.55rem', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)' }}>
          {frame && frame.imageUrl ? (
            <img
              src={frame.imageUrl}
              alt={`Scene ${sceneNum} Master Frame`}
              style={{ width: '58px', height: '40px', objectFit: 'cover', borderRadius: '5px', border: '1px solid rgba(52, 211, 153, 0.6)' }}
            />
          ) : (
            <div style={{ width: '58px', height: '40px', borderRadius: '5px', background: '#11151f', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.58rem', color: 'var(--text-dim)', border: '1px dashed rgba(255,255,255,0.2)' }}>
              3.{sceneNum}
            </div>
          )}
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <div style={{ fontSize: '0.64rem', display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#cbd2de', fontWeight: 600 }}>🖼️ 3.{sceneNum} Master Frame:</span>
              <span style={{ color: frame && frame.imageUrl ? '#34d399' : '#f59e0b', fontWeight: 700 }}>
                {frame && frame.imageUrl ? '✓ Linked' : '⏳ Pending'}
              </span>
            </div>
            <div style={{ fontSize: '0.62rem', color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              📜 {sceneScript?.voiceover_english || sceneScript?.raw_prompt || 'Waiting for Scene Script...'}
            </div>
          </div>
        </div>

        {/* Live Progress Bar for 4.K Video Synthesis */}
        {isThisVideoRendering && data.liveProgress && (
          <div style={{ padding: '0.42rem 0.55rem', background: 'rgba(244, 63, 94, 0.14)', borderRadius: '6px', border: '1px solid rgba(244, 63, 94, 0.4)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.66rem', color: '#fda4af', fontWeight: 600, marginBottom: '3px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Loader2 size={11} className="animate-spin" />
                {data.liveProgress.message || `Synthesizing 4.${sceneNum} Video (${wanModel})...`}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800 }}>
                {data.liveProgress.percentage || 50}%
              </span>
            </div>
            <div style={{ height: '5px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${data.liveProgress.percentage || 50}%`,
                  background: 'linear-gradient(90deg, #f43f5e, #f59e0b)',
                  transition: 'width 0.25s ease',
                }}
              />
            </div>
          </div>
        )}

        {/* Generated Scene Video Player */}
        {videoClip && videoClip.videoUrl ? (
          <div style={{ background: '#05070b', padding: '0.5rem', borderRadius: '7px', border: '1px solid rgba(244, 63, 94, 0.35)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem', fontSize: '0.65rem' }}>
              <span style={{ color: '#fda4af', fontWeight: 700 }}>
                🎬 Scene {sceneNum} Video ({videoClip.wan_model || wanModel})
              </span>
              <a
                href={videoClip.videoUrl}
                download={`scene_${sceneNum}_${wanModel}.mp4`}
                target="_blank"
                rel="noreferrer"
                style={{ color: '#34d399', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '3px', fontWeight: 600 }}
              >
                <Download size={11} /> Download MP4
              </a>
            </div>
            <video
              key={videoClip.videoUrl}
              src={videoClip.videoUrl}
              controls
              loop
              playsInline
              onPlay={() => {
                if (sceneScript?.voiceover_english) {
                  speakVoiceOver(sceneScript.voiceover_english);
                }
              }}
              style={{ width: '100%', maxHeight: '220px', borderRadius: '6px', background: '#000', objectFit: 'cover' }}
            />
            <div style={{ marginTop: '0.35rem', fontSize: '0.63rem', color: '#cbd2de', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '6px' }}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                🎙️ <strong>{sceneScript?.voiceover_speaker || 'Speaker'}:</strong> "{sceneScript?.voiceover_english || ''}"
              </span>
              <button
                type="button"
                className="planner-reopt-all-btn nodrag"
                style={{ flexShrink: 0 }}
                onClick={() => speakVoiceOver(sceneScript?.voiceover_english)}
              >
                <Volume2 size={10} /> VO
              </button>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: '0.69rem', background: '#0c0f16', padding: '0.85rem', borderRadius: '7px', border: '1px dashed rgba(244, 63, 94, 0.35)', textAlign: 'center', color: 'var(--text-dim)' }}>
            {isThisVideoRendering ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.35rem', color: '#fda4af' }}>
                <Loader2 size={20} className="animate-spin" />
                <span>Synthesizing Scene {sceneNum} Video from Card 3.{sceneNum} ({wanModel})...</span>
              </div>
            ) : bothReady ? (
              `✓ Card 3.${sceneNum} (Master Frame + Scene ${sceneNum} Script) is ready! Click below to synthesize Scene ${sceneNum} Video with ${wanModel}.`
            ) : (
              `Waiting for Card 3.${sceneNum} Master Frame Image & Scene ${sceneNum} Script...`
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.15rem' }}>
          <button
            type="button"
            className="card-action-btn nodrag"
            onClick={(e) => {
              e.stopPropagation();
              if (data.onGenerateSceneVideo) data.onGenerateSceneVideo(sceneIdx);
            }}
            disabled={!bothReady || data.isGeneratingVideos || data.isRegeneratingVideo === sceneIdx}
          >
            {data.isRegeneratingVideo === sceneIdx ? (
              <>
                <Loader2 size={12} className="animate-spin" />
                <span>Rendering 4.{sceneNum}...</span>
              </>
            ) : (
              <>
                <RefreshCw size={12} />
                <span>{videoClip ? `Regenerate 4.${sceneNum} Video` : `Generate 4.${sceneNum} Video`}</span>
              </>
            )}
          </button>

          <button
            type="button"
            className="card-action-btn nodrag"
            style={{ background: 'rgba(244, 63, 94, 0.2)', borderColor: '#f43f5e', color: '#fda4af' }}
            onClick={(e) => {
              e.stopPropagation();
              if (data.onGenerateAllVideos) data.onGenerateAllVideos();
            }}
            disabled={!bothReady || data.isGeneratingVideos}
          >
            {data.isGeneratingVideos ? (
              <>
                <Loader2 size={12} className="animate-spin" />
                <span>Rendering All...</span>
              </>
            ) : (
              <>
                <Play size={12} fill="currentColor" />
                <span>Generate All Videos ({wanModel})</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
});


