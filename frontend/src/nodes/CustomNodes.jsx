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

export const NodeModelSelector = memo(({ data, isConnectable }) => {
  const selectedModel = data.fluxModel || 'Qwen-Image-2.1';
  const aspectRatio = data.aspectRatio || '16:9';
  const activeOption = IMAGE_MODEL_OPTIONS.find((m) => m.id === selectedModel) || IMAGE_MODEL_OPTIONS[0];

  return (
    <div className={`studio-node model-selector-node accent-sky ${data.isActive ? 'active-processing' : ''}`}>
      <div className="studio-node-header">
        <div className="studio-node-title">
          <Cpu size={16} style={{ color: '#38bdf8' }} />
          <span>0. Image Generation Model</span>
        </div>
        <span
          className="studio-node-badge"
          style={{ color: '#38bdf8', borderColor: 'rgba(56, 189, 248, 0.4)' }}
        >
          Select Engine First
        </span>
      </div>

      <div className="studio-node-body">
        <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', lineHeight: 1.4 }}>
          Select which local <strong>Image Generation Model</strong> from <code style={{ color: '#38bdf8' }}>backend/models/</code> will synthesize your short film frames before entering your scenario:
        </div>

        {/* Model Option Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
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
                    <span style={{ fontWeight: 700, fontSize: '0.75rem', color: '#ffffff' }}>
                      {opt.title}
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: '0.58rem',
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

                <div style={{ fontSize: '0.65rem', color: '#94a3b8', lineHeight: 1.35 }}>
                  {opt.desc}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2px', fontSize: '0.6rem', fontFamily: 'var(--font-mono)', color: isSelected ? '#7dd3fc' : '#64748b' }}>
                  <span>📂 {opt.path}</span>
                  <span>{isSelected ? '✓ ACTIVE' : 'Click to use'}</span>
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

        {/* Active Hardware & Model Summary Box */}
        <div style={{ background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.3)', borderRadius: '7px', padding: '0.45rem 0.6rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.66rem' }}>
          <span style={{ color: '#cbd2de' }}>
            Selected: <strong style={{ color: '#38bdf8' }}>{activeOption.id}</strong>
          </span>
          <span style={{ color: '#34d399', fontWeight: 700 }}>➔ Feeds Card 1 & Card 3</span>
        </div>
      </div>

      <Handle type="source" position={Position.Right} isConnectable={isConnectable} />
    </div>
  );
});

// ============================================================================
// NODE 1: Scenario & Character Input Card
// ============================================================================
export const NodePrompt = memo(({ data, isConnectable }) => {
  const scenes = data.scenes || [{ id: 1, text: '', duration: '5s' }];
  const selectedChars = data.characters || ['male', 'female'];
  const fluxModel = data.fluxModel || 'Qwen-Image-2.1';

  const hasMale = selectedChars.includes('male');
  const hasFemale = selectedChars.includes('female');

  return (
    <div className={`studio-node prompt-node accent-purple ${data.isActive ? 'active-processing' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />
      <div className="studio-node-header">
        <div className="studio-node-title">
          <Sparkles size={16} style={{ color: '#a855f7' }} />
          <span>1. Scenario & Character Input</span>
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
              title="Clear loaded history and start a fresh New Generation"
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
        {/* Active Model & Scene Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', paddingBottom: '0.45rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.71rem', color: 'var(--text-dim)', fontWeight: 600 }}>Image Model (Card 0):</span>
            <span style={{ fontSize: '0.68rem', color: '#38bdf8', fontWeight: 700, background: 'rgba(56, 189, 248, 0.12)', padding: '2px 8px', borderRadius: '5px', border: '1px solid rgba(56, 189, 248, 0.35)' }}>
              ⚡ {fluxModel}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.71rem', color: 'var(--text-dim)', fontWeight: 600 }}>Scenes Count:</span>
            <div className="selector-chip-group">
              {[1, 2, 3].map((cnt) => (
                <button
                  key={cnt}
                  type="button"
                  className={`selector-chip nodrag ${scenes.length === cnt ? 'active' : ''}`}
                  onClick={() => data.onSetSceneCount && data.onSetSceneCount(cnt)}
                >
                  {cnt} {cnt === 1 ? 'Scene' : 'Scenes'}
                </button>
              ))}
            </div>
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

        {/* Turnaround Character Reference Selector */}
        <div style={{ paddingBottom: '0.45rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Users size={12} style={{ color: '#a855f7' }} />
              Character Turnaround Reference:
            </span>
            <span style={{ fontSize: '0.62rem', color: '#d8b4fe' }}>Click to toggle</span>
          </div>

          <div className="char-ref-strip">
            <div
              className={`char-ref-card nodrag ${hasMale ? 'selected' : ''}`}
              onClick={() => data.onToggleCharacter && data.onToggleCharacter('male')}
              title="Toggle Male Lead Character from 360° Turnaround Sheet"
            >
              <img
                src="/data/characters/male_turnaround.png"
                alt="Male Lead"
                className="char-ref-thumb"
              />
              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#fff' }}>Male Lead</div>
                <div style={{ fontSize: '0.6rem', color: hasMale ? '#d8b4fe' : 'var(--text-dim)' }}>
                  {hasMale ? '✓ Active in Prompt' : 'Click to include'}
                </div>
              </div>
            </div>

            <div
              className={`char-ref-card nodrag ${hasFemale ? 'selected' : ''}`}
              onClick={() => data.onToggleCharacter && data.onToggleCharacter('female')}
              title="Toggle Female Lead Character from Studio Turnaround Sheet"
            >
              <img
                src="/data/characters/female_turnaround.png"
                alt="Female Lead"
                className="char-ref-thumb"
              />
              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#fff' }}>Female Lead</div>
                <div style={{ fontSize: '0.6rem', color: hasFemale ? '#d8b4fe' : 'var(--text-dim)' }}>
                  {hasFemale ? '✓ Active in Prompt' : 'Click to include'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Dynamic Scenario Input List */}
        <div className="scenes-scroll-list">
          {scenes.map((scene, idx) => (
            <div key={scene.id || idx} className="scene-card-item">
              <div className="scene-card-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <span className="scene-num-badge">S{idx + 1}</span>
                  <span style={{ fontWeight: 600, color: 'var(--text-main)', fontSize: '0.72rem' }}>
                    Scenario {idx + 1} (English / Singlish)
                  </span>
                </div>
              </div>

              <textarea
                className="node-textarea nodrag nowheel"
                value={scene.text || ''}
                onChange={(e) => data.onChangeSceneText && data.onChangeSceneText(idx, e.target.value)}
                onMouseDown={(e) => e.stopPropagation()}
                placeholder={`Type your short scenario here (e.g. "Two characters meeting at a cozy rain-streaked coffee shop at night" or "kollekui kellekui coffee shop ekaka katha krnwa")...`}
              />
            </div>
          ))}
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
                <span>Optimizing Prompt (Card 2)...</span>
              </>
            ) : (
              <>
                <Wand2 size={14} />
                <span>Step 1: Optimize Prompt Only ➔ Card 2</span>
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
                <span>Running FLUX GPU Pipeline...</span>
              </>
            ) : (
              <>
                <Play size={14} fill="currentColor" />
                <span>Optimize & Generate Image ({fluxModel})</span>
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
// NODE 2: AI Prompt Optimizer Agent (Scenario -> FLUX Master Prompt)
// ============================================================================
export const NodePlanner = memo(({ data, isConnectable }) => {
  const [selectedSceneIdx, setSelectedSceneIdx] = useState(0);

  const optimizedScenes = data.optimizedScenes || [];
  const currentScene = optimizedScenes[selectedSceneIdx] || (optimizedScenes.length > 0 ? optimizedScenes[0] : null);

  return (
    <div className={`studio-node planner-node accent-cyan ${data.isActive ? 'active-processing' : ''}`}>
      <Handle type="target" position={Position.Left} isConnectable={isConnectable} />
      <div className="studio-node-header">
        <div className="studio-node-title">
          <BrainCircuit size={16} style={{ color: '#06b6d4' }} />
          <span>2. AI Prompt Optimizer Agent</span>
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
              title="Re-optimize all scenarios into fresh FLUX prompts"
            >
              {data.isOptimizing ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
              <span>Re-optimize</span>
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
            {optimizedScenes.length > 1 && (
              <div style={{ display: 'flex', gap: '0.35rem', overflowX: 'auto', paddingBottom: '0.2rem' }}>
                {optimizedScenes.map((sc, idx) => (
                  <button
                    key={sc.id || idx}
                    type="button"
                    className={`planner-tab-btn nodrag ${selectedSceneIdx === idx ? 'active' : ''}`}
                    onClick={() => setSelectedSceneIdx(idx)}
                  >
                    Scene {idx + 1}
                  </button>
                ))}
              </div>
            )}

            {currentScene && (
              <div className="optimized-scene-display">
                <div className="raw-prompt-preview">
                  <span className="raw-label">
                    <Sparkles size={10} /> User Scenario (Card 1):
                  </span>
                  <span>"{currentScene.raw_prompt || '(Default scenario)'}"</span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.15rem' }}>
                  <span style={{ fontSize: '0.68rem', color: '#67e8f9', fontWeight: 700, textTransform: 'uppercase' }}>
                    ✨ Optimized {data.fluxModel || 'FLUX'} Master Prompt (Editable):
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
                    data.onEditOptimizedPrompt(selectedSceneIdx, e.target.value)
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
                      if (data.onRegenerateScene) data.onRegenerateScene(selectedSceneIdx);
                    }}
                    disabled={data.isRegeneratingScene === selectedSceneIdx || data.isOptimizing}
                  >
                    {data.isRegeneratingScene === selectedSceneIdx ? (
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
                    <span>Generate Image ➔ Card 3</span>
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div style={{ fontSize: '0.73rem', background: '#12151c', padding: '1rem', borderRadius: '8px', border: '1px solid var(--border-subtle)', textAlign: 'center', color: 'var(--text-dim)' }}>
            <BrainCircuit size={26} style={{ margin: '0 auto 0.45rem', opacity: 0.65, color: '#06b6d4' }} />
            <div>Type a scenario in <strong>Card 1</strong> and click <strong>Optimize Prompt</strong> to generate a cinema-grade {data.fluxModel || 'FLUX.1'} prompt with Character Consistency.</div>
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

export function enrichSceneWithVideoAndVO(sc = {}, idx = 0, forceFresh = false) {
  const rawText = (sc.raw_prompt || sc.action || sc.text || '').trim();
  const rawLower = (rawText + ' ' + (sc.environment || '') + ' ' + (sc.optimized_image_prompt || '')).toLowerCase();

  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  const camSpec = pick(CAMERA_VARIATIONS);
  const camMotion = forceFresh ? camSpec.motion : (sc.camera_motion || camSpec.motion);
  const camLens = forceFresh ? camSpec.cam : (sc.camera || camSpec.cam);

  let defaultMotion = '';
  let defaultSpeaker = idx % 2 === 0 ? 'Arjun (Male Lead)' : 'Maya (Female Lead)';
  let defaultVoSi = '';
  let defaultVoEn = '';
  let defaultSfx = 'Natural cinema room tone & soft emotional background score';

  if (rawLower.includes('coffee') || rawLower.includes('cafe') || rawLower.includes('tea')) {
    const v = pick([
      {
        speaker: 'Arjun (Male Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Warm wisps of steam curl upward from the ceramic coffee cups while soft raindrops trickle down the cafe glass behind them. Arjun and Maya maintain warm eye contact, smiling naturally and speaking softly across the wooden table at 24fps. (${rawText})`,
        si: 'සමහර හමුවීම් තියෙනවා... වචන ගොඩක් නැතුවත් හිතටම දැනෙන. (Samahara hamuweem thiyenawa, wachana godak nathuwath hithatama danena.)',
        en: "Some meetings don't need many words... they quietly speak straight to the heart.",
      },
      {
        speaker: 'Maya (Female Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Maya gently wraps her fingers around the warm coffee cup, looking up at Arjun with a soft, genuine smile while amber pendant lights and city bokeh shimmer through the rainy cafe window at 24fps. (${rawText})`,
        si: 'වැස්ස වැටෙන මේ නිහඬ හවසේ, ඔයාගේ කතාව අහන් ඉන්න එකම ලොකු සැනසීමක්. (Wassa watena me nihada hawase, oyage kathawa ahan inna ekama loku sanaseemak.)',
        en: 'On this quiet rainy evening, just listening to your voice feels like home.',
      },
      {
        speaker: 'Arjun & Maya (Dialogue)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Both Arjun and Maya share a candid, heartfelt laugh across the wooden cafe table as rising coffee steam and soft window rain create a rich 24fps cinematic atmosphere. (${rawText})`,
        si: 'කෝපි කෝප්පයක උණුසුමත් එක්ක, අපේ අලුත් කතාව මෙතනින් පටන් ගත්තා. (Kopi koppayaka unusumath ekka, ape aluth kathawa methanin patan gaththa.)',
        en: 'Over the warmth of a simple cup of coffee, a brand new chapter of our story began.',
      },
    ]);
    defaultMotion = v.motion;
    defaultSpeaker = v.speaker;
    defaultVoSi = v.si;
    defaultVoEn = v.en;
    defaultSfx = 'Soft rain tapping on cafe window glass, gentle acoustic guitar notes, subtle coffee cup clink';
  } else if (rawLower.includes('rain') || rawLower.includes('wassa') || rawLower.includes('wehi') || rawLower.includes('sidewalk') || rawLower.includes('street')) {
    const v = pick([
      {
        speaker: 'Maya (Female Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Glistening raindrops fall steadily in slow motion as glowing neon reflections shimmer on the wet pavement. Arjun and Maya walk side by side along the street, turning toward each other with a warm smile at 24fps. (${rawText})`,
        si: 'වැස්සත් එක්ක ආපු ඒ රාත්‍රිය, අපේ කතාවේ ලස්සනම පිටුවක් වුණා. (Wassath ekka aapu e rathriya, ape kathawe lassanama pituwak wuna.)',
        en: 'That rainy night in the city became the most unforgettable page of our story.',
      },
      {
        speaker: 'Arjun (Male Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Ripples spread across neon-lit puddles in the rain as Arjun glances at Maya with an expressive, protective smile while walking through the glowing night street at 24fps. (${rawText})`,
        si: 'හැම වැහි බිඳුවක්ම කියන්නේ අපි දෙන්නාගේ හිතේ තිබුණු නිහඬ හැඟීමක්. (Hama wahi binduwakma kiyanne api dennage hithe thibunu nihada hageemak.)',
        en: 'Every falling raindrop echoed the quiet feelings we never said out loud.',
      },
    ]);
    defaultMotion = v.motion;
    defaultSpeaker = v.speaker;
    defaultVoSi = v.si;
    defaultVoEn = v.en;
    defaultSfx = 'Steady city rainfall on wet pavement, distant soft traffic hum, gentle piano melody';
  } else if (rawLower.includes('mountain') || rawLower.includes('kanda') || rawLower.includes('hill') || rawLower.includes('mist')) {
    const v = pick([
      {
        speaker: 'Maya (Female Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Soft golden-hour mist drifts slowly across the lush green mountain hills in the background as a gentle breeze moves their hair; Arjun and Maya look at each other with a peaceful smile at 24fps. (${rawText})`,
        si: 'මීදුම අතරින් පේන මේ කඳු යාය වගේම, ජීවිතයත් හරිම නිස්කලංකයි. (Meeduma atharin pena me kandu yaya wagema, jeewithayath harima niskalankai.)',
        en: 'Up here above the drifting mountain mist, time finally slows down for us.',
      },
      {
        speaker: 'Arjun (Male Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Standing above the drifting highland clouds, warm morning sunlight catches their faces as Arjun and Maya exchange a heartfelt glance at 24fps. (${rawText})`,
        si: 'මේ නිහඬ කඳු මුදුනේදී, හිතට දැනෙන්නේ පුදුමාකාර නිදහසක්. (Me nihada kandu mudunedi, hithata danenne pudumakara nidahasak.)',
        en: 'Standing above the quiet clouds, everything finally feels clear and free.',
      },
    ]);
    defaultMotion = v.motion;
    defaultSpeaker = v.speaker;
    defaultVoSi = v.si;
    defaultVoEn = v.en;
    defaultSfx = 'Soft mountain wind breeze, distant morning birdsong, swelling cinematic strings';
  } else if (rawLower.includes('beach') || rawLower.includes('muhuda') || rawLower.includes('ocean') || rawLower.includes('sunset')) {
    const v = pick([
      {
        speaker: 'Arjun & Maya (Narrator)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Gentle turquoise ocean waves roll onto the wet golden sand while a warm sunset breeze stirs their hair as Arjun and Maya walk slowly along the shoreline at 24fps. (${rawText})`,
        si: 'ඉර බැහැගෙන යන හැම හවසකම, අලුත් බලාපොරොත්තුවක් හිතේ ඉතුරු වෙනවා. (Ira bahagena yana hama hawasakama, aluth balaporoththuwak hithe ithuru wenawa.)',
        en: 'With every golden sunset by the ocean, a quiet new hope stays behind.',
      },
      {
        speaker: 'Maya (Female Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Golden sunset reflections shimmer on the ocean tide as Maya turns toward Arjun with a radiant, emotional smile along the breezy shoreline at 24fps. (${rawText})`,
        si: 'මුහුදු රළ වගේම සමහර මතකයන් කවදාවත් හිතෙන් මැකෙන්නේ නෑ. (Muhudu rala wagema samahara mathakayan kawadawath hithen makenne naa.)',
        en: 'Like the endless ocean tide, some memories never fade from the heart.',
      },
    ]);
    defaultMotion = v.motion;
    defaultSpeaker = v.speaker;
    defaultVoSi = v.si;
    defaultVoEn = v.en;
    defaultSfx = 'Gentle ocean waves washing ashore, soft coastal breeze, warm ambient guitar';
  } else {
    const v = pick([
      {
        speaker: 'Arjun (Male Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Starting from the Scene ${idx + 1} Master Frame: natural 24fps human micro-expressions, subtle eye contact, realistic breathing and hair/fabric movement matching "${rawText || 'cinematic scene'}".`,
        si: `මේ මොහොත අපේ කතාවේ කවදාවත් අමතක නොවන හැරවුම් ලක්ෂ්‍යයක් වුණා. (${rawText || 'Short Film Scene'})`,
        en: `This very moment became a turning point we would remember forever — ${rawText || 'cinematic short film scene'}.`,
      },
      {
        speaker: 'Maya (Female Lead)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Smooth 24fps cinematic movement as Arjun and Maya interact naturally in "${rawText || 'cinematic scene'}", with expressive head turns and shallow depth-of-field parallax.`,
        si: `හිතේ තිබුණු හැම බලාපොරොත්තුවක්ම සැබෑ වුණු ඒ සුවිශේෂී මොහොත... (${rawText || 'Short Film Scene'})`,
        en: `Every quiet hope we carried finally came alive in this moment — ${rawText || 'cinematic short film scene'}.`,
      },
      {
        speaker: 'Arjun & Maya (Narrator)',
        motion: `[Wan 2.2 I2V Motion Script — Scene ${idx + 1}] ${camMotion}. Dynamic 24fps short-film motion capturing "${rawText || 'cinematic scene'}" with lifelike posture shifts, warm emotional gaze, and atmospheric background motion.`,
        si: `ජීවිතයේ සමහර තත්පර කිහිපයක් මුළු කතාවම වෙනස් කරනවා. (${rawText || 'Short Film Scene'})`,
        en: `Sometimes just a few seconds can change the entire story — ${rawText || 'cinematic short film scene'}.`,
      },
    ]);
    defaultMotion = v.motion;
    defaultSpeaker = v.speaker;
    defaultVoSi = v.si;
    defaultVoEn = v.en;
  }

  let optImgPrompt = sc.optimized_image_prompt || '';
  if (forceFresh && rawText) {
    const stock = pick([
      'ARRI Alexa 65 with Kodak Vision3 500T color science',
      'RED V-Raptor 8K VV with Cooke Anamorphic/i Full Frame optics',
      'Sony VENICE 2 8K with Panavision Primo 70 cinema lens',
    ]);
    optImgPrompt = `Cinematic 8K short film still shot on ${stock}, ${camLens}. Scene scenario & action: ${rawText}. ${sc.character_description || 'Featuring consistent lead characters arjun_male and maya_female'}. Setting: ${sc.environment || rawText}. Lighting: ${sc.lighting || 'naturalistic Hollywood cinema lighting with warm rim light'}. Ultra-realistic skin micro-texture, natural expressive eyes, masterpiece photorealism.`;
  }

  return {
    ...sc,
    camera: camLens,
    camera_motion: camMotion,
    optimized_image_prompt: optImgPrompt || sc.optimized_image_prompt,
    video_motion_prompt: (!forceFresh && sc.video_motion_prompt && sc.video_motion_prompt.trim()) ? sc.video_motion_prompt : defaultMotion,
    voiceover_speaker: (!forceFresh && sc.voiceover_speaker && sc.voiceover_speaker.trim()) ? sc.voiceover_speaker : defaultSpeaker,
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

