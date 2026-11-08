import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
} from '@xyflow/react';

import {
  NodeModelSelector,
  NodePrompt,
  NodePlanner,
  NodeKeyframeGenerator,
  NodeVideoScriptStudio,
  enrichSceneWithVideoAndVO,
} from './nodes/CustomNodes';

import {
  Play,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Terminal,
  ExternalLink,
  Sparkles,
  Loader2,
  History,
  Check,
  Clock,
  Cpu,
  PlusCircle
} from 'lucide-react';

const nodeTypes = {
  modelSelectorNode: NodeModelSelector,
  promptNode: NodePrompt,
  plannerNode: NodePlanner,
  keyframeNode: NodeKeyframeGenerator,
  videoScriptNode: NodeVideoScriptStudio,
};

const DEFAULT_NODES = [
  {
    id: '0',
    type: 'modelSelectorNode',
    position: { x: 20, y: 210 },
    data: {
      fluxModel: 'Qwen-Image-2.1',
      aspectRatio: '16:9',
    },
  },
  {
    id: '1',
    type: 'promptNode',
    position: { x: 455, y: 210 },
    data: {
      fluxModel: 'Qwen-Image-2.1',
      aspectRatio: '16:9',
      characters: ['male', 'female'],
      scenes: [
        {
          id: 1,
          text: 'Two characters meeting at a cozy rain-streaked coffee shop window at night, talking warmly over steaming coffee',
          duration: '5s',
        },
      ],
    },
  },
  {
    id: '2',
    type: 'plannerNode',
    position: { x: 910, y: 210 },
    data: {
      fluxModel: 'Qwen-Image-2.1',
      aspectRatio: '16:9',
      optimizedScenes: [],
    },
  },
  {
    id: '3',
    type: 'keyframeNode',
    position: { x: 1385, y: 35 },
    data: {
      fluxModel: 'Qwen-Image-2.1',
      frames: [],
    },
  },
  {
    id: '4',
    type: 'videoScriptNode',
    position: { x: 1385, y: 500 },
    data: {
      aspectRatio: '16:9',
      optimizedScenes: [],
    },
  },
];

const DEFAULT_EDGES = [
  {
    id: 'e0-1',
    source: '0',
    target: '1',
    animated: true,
    style: {
      stroke: '#38bdf8',
      strokeWidth: 2.6,
      filter: 'drop-shadow(0 0 6px rgba(56, 189, 248, 0.75))',
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#38bdf8' },
  },
  {
    id: 'e1-2',
    source: '1',
    target: '2',
    animated: true,
    style: {
      stroke: '#a855f7',
      strokeWidth: 2.6,
      filter: 'drop-shadow(0 0 6px rgba(168, 85, 247, 0.75))',
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#a855f7' },
  },
  {
    id: 'e2-3',
    source: '2',
    target: '3',
    animated: true,
    style: {
      stroke: '#10b981',
      strokeWidth: 2.6,
      filter: 'drop-shadow(0 0 6px rgba(16, 185, 129, 0.75))',
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#10b981' },
  },
  {
    id: 'e2-4',
    source: '2',
    target: '4',
    animated: true,
    style: {
      stroke: '#f59e0b',
      strokeWidth: 2.6,
      filter: 'drop-shadow(0 0 6px rgba(245, 158, 11, 0.75))',
    },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#f59e0b' },
  },
];

const STORAGE_KEY = 'film_agent_flux_nodes_v5';

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length === DEFAULT_NODES.length) {
          return DEFAULT_NODES.map((dNode) => {
            const found = parsed.find((p) => p.id === dNode.id);
            return found ? { ...dNode, position: found.position } : dNode;
          });
        }
      }
    } catch {}
    return DEFAULT_NODES;
  });

  const [edges, setEdges, onEdgesChange] = useEdgesState(DEFAULT_EDGES);

  const [isRunning, setIsRunning] = useState(false);
  const [isHeaderCollapsed, setIsHeaderCollapsed] = useState(false);
  const [isConsoleMinimized, setIsConsoleMinimized] = useState(false);
  const [gpuInfo, setGpuInfo] = useState('RTX 4060 (8GB VRAM)');
  const [logs, setLogs] = useState([
    { id: 1, time: new Date().toLocaleTimeString(), text: '[System] FILM AGENT — FLUX.1 / FLUX.2 Studio initialized.', type: 'info' },
    { id: 2, time: new Date().toLocaleTimeString(), text: '[Card 1] Character Turnaround Reference Sheets (Male & Female Leads) loaded.', type: 'success' },
    { id: 3, time: new Date().toLocaleTimeString(), text: '[Card 2] AI Prompt Optimizer Agent (T5-XXL Natural Language) ready.', type: 'success' },
    { id: 4, time: new Date().toLocaleTimeString(), text: '[Card 3] Local GPU Engine (NVIDIA GeForce RTX 4060 CUDA) connected.', type: 'info' },
  ]);

  const logsEndRef = useRef(null);
  const runWorkflowRef = useRef(null);
  const optimizeOnlyRef = useRef(null);
  const generateFromOptimizedRef = useRef(null);
  const [liveProgress, setLiveProgress] = useState(null);
  const progressPollRef = useRef(null);

  const [historyRuns, setHistoryRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const historyRef = useRef(null);

  const addLog = useCallback((text, type = 'info') => {
    setLogs((prev) => [
      ...prev,
      { id: Date.now() + Math.random(), time: new Date().toLocaleTimeString(), text, type },
    ]);
  }, []);

  useEffect(() => {
    if (!isConsoleMinimized && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isConsoleMinimized]);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (historyRef.current && !historyRef.current.contains(e.target)) {
        setIsHistoryOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch GPU status & History on mount
  const fetchHistoryList = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/history');
      if (res.ok) {
        const list = await res.json();
        setHistoryRuns(list);
      }
    } catch {}
  }, []);

  useEffect(() => {
    fetchHistoryList();
    fetch('/api/v1/generate/image/status')
      .then((r) => (r.ok ? r.json() : null))
      .then((st) => {
        if (st && st.gpu_device) {
          setGpuInfo(`${st.gpu_device} (${st.vram_gb}GB)`);
        }
      })
      .catch(() => {});
  }, [fetchHistoryList]);

  const onNodeDragStop = useCallback(
    (_, node) => {
      setNodes((currentNodes) => {
        try {
          const positions = currentNodes.map((n) => ({ id: n.id, position: n.position }));
          localStorage.setItem(STORAGE_KEY, JSON.stringify(positions));
        } catch {}
        return currentNodes;
      });
    },
    [setNodes]
  );

  const resetLayout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setNodes((nds) =>
      nds.map((n) => {
        const def = DEFAULT_NODES.find((d) => d.id === n.id);
        return def ? { ...n, position: def.position } : n;
      })
    );
    addLog('Canvas layout reset to default 3-node workflow.', 'info');
  }, [setNodes, addLog]);

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  // Start a Fresh New Generation (Clears loaded History Run & resets Cards 1, 2, 3, 4)
  const startNewGeneration = useCallback(() => {
    setSelectedRunId(null);
    setIsHistoryOpen(false);
    setNodes((nds) => {
      const modelNode = nds.find((n) => n.id === '0');
      const currentModel = modelNode?.data?.fluxModel || 'Qwen-Image-2.1';
      const currentRatio = modelNode?.data?.aspectRatio || '16:9';

      return nds.map((node) => {
        if (node.id === '0') {
          return {
            ...node,
            data: {
              ...node.data,
              fluxModel: currentModel,
              aspectRatio: currentRatio,
            },
          };
        }
        if (node.id === '1') {
          return {
            ...node,
            data: {
              ...node.data,
              fluxModel: currentModel,
              aspectRatio: currentRatio,
              characters: ['male', 'female'],
              scenes: [{ id: 1, text: '', duration: '5s' }],
              isActive: false,
              isOptimizing: false,
            },
          };
        }
        if (node.id === '2') {
          return {
            ...node,
            data: {
              ...node.data,
              fluxModel: currentModel,
              aspectRatio: currentRatio,
              optimizedScenes: [],
              isActive: false,
              isOptimizing: false,
            },
          };
        }
        if (node.id === '3') {
          return {
            ...node,
            data: {
              ...node.data,
              fluxModel: currentModel,
              runId: null,
              folderPath: null,
              frames: [],
              isActive: false,
            },
          };
        }
        if (node.id === '4') {
          return {
            ...node,
            data: {
              ...node.data,
              aspectRatio: currentRatio,
              runId: null,
              optimizedScenes: [],
              isActive: false,
            },
          };
        }
        return node;
      });
    });
    addLog('[New Generation] Cleared previous history run. Ready for a fresh scenario in Card 1!', 'success');
  }, [setNodes, addLog]);

  // Restore a History Run into Cards 1, 2, 3, 4
  const loadHistoryRun = useCallback(
    async (runId, keepCurrentModel = false) => {
      try {
        addLog(`[History] Restoring run: ${runId}...`, 'info');
        const res = await fetch(`/api/v1/history/${runId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        setSelectedRunId(runId);
        setIsHistoryOpen(false);

        setNodes((nds) =>
          nds.map((node) => {
            const targetModel = keepCurrentModel ? node.data.fluxModel : (data.flux_model || node.data.fluxModel);
            if (node.id === '0') {
              return {
                ...node,
                data: {
                  ...node.data,
                  fluxModel: targetModel,
                  aspectRatio: data.aspect_ratio || node.data.aspectRatio,
                },
              };
            }
            if (node.id === '1') {
              const restoredScenes =
                data.storyboard?.scenes?.length > 0
                  ? data.storyboard.scenes
                  : (data.planned_scenes || []).map((ps, idx) => ({
                      id: idx + 1,
                      text: ps.raw_prompt || '',
                      duration: ps.duration || '5s',
                    }));
              return {
                ...node,
                data: {
                  ...node.data,
                  fluxModel: targetModel,
                  aspectRatio: data.aspect_ratio || node.data.aspectRatio,
                  scenes: restoredScenes.length > 0 ? restoredScenes : node.data.scenes,
                },
              };
            }
            if (node.id === '2') {
              const enriched = (data.planned_scenes || []).map((ps, idx) => enrichSceneWithVideoAndVO(ps, idx));
              return {
                ...node,
                data: {
                  ...node.data,
                  fluxModel: targetModel,
                  aspectRatio: data.aspect_ratio || node.data.aspectRatio,
                  optimizedScenes: enriched,
                },
              };
            }
            if (node.id === '3') {
              return {
                ...node,
                data: {
                  ...node.data,
                  fluxModel: targetModel,
                  runId: data.run_id,
                  folderPath: data.folder_path || `data/keyframes/${data.run_id}`,
                  frames: (data.keyframes || []).map((f) => ({
                    ...f,
                    imageUrl: f.imageUrl || f.url,
                  })),
                },
              };
            }
            if (node.id === '4') {
              const enriched = (data.planned_scenes || []).map((ps, idx) => enrichSceneWithVideoAndVO(ps, idx));
              return {
                ...node,
                data: {
                  ...node.data,
                  aspectRatio: data.aspect_ratio || node.data.aspectRatio,
                  runId: data.run_id,
                  optimizedScenes: enriched,
                },
              };
            }
            return node;
          })
        );
        addLog(`[History] Restored "${data.title || runId}" into Cards 1, 2, 3, and 4.`, 'success');
      } catch (err) {
        addLog(`[History] Failed to restore run: ${err.message}`, 'error');
      }
    },
    [addLog, setNodes]
  );

  const didAutoRestoreRef = useRef(false);
  useEffect(() => {
    fetch('/api/v1/history')
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => {
        if (Array.isArray(list) && list.length > 0 && !didAutoRestoreRef.current) {
          didAutoRestoreRef.current = true;
          loadHistoryRun(list[0].run_id, true);
        }
      })
      .catch(() => {});
  }, [loadHistoryRun]);

  // Step 1: Optimize Scenario(s) from Card 1 -> Card 2 & Card 4 (Always generates a fresh optimization!)
  const onOptimizeAllScenes = useCallback(async () => {
    const promptNode = nodesRef.current.find((n) => n.id === '1');
    const plannerNode = nodesRef.current.find((n) => n.id === '2');
    const prevScenes = plannerNode?.data?.optimizedScenes || [];

    const scenes = promptNode?.data?.scenes || [{ id: 1, text: '', duration: '5s' }];
    const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
    const fluxModel = promptNode?.data?.fluxModel || 'Qwen-Image-2.1';
    const characters = promptNode?.data?.characters || ['male', 'female'];

    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === '1') return { ...n, data: { ...n.data, isOptimizing: true, isActive: true } };
        if (n.id === '2') return { ...n, data: { ...n.data, isOptimizing: true, isActive: true, optimizedScenes: [] } };
        if (n.id === '4') return { ...n, data: { ...n.data, isActive: true, optimizedScenes: [] } };
        return n;
      })
    );

    addLog(`[Card 1 ➔ Card 2 & Card 4] Generating NEW ${fluxModel} Master Prompt(s) + NEW Video Motion & Voice-Over Script(s)...`, 'info');

    let optimizedScenesData = [];
    try {
      const planRes = await fetch('/api/v1/scenes/plan-multi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          flux_model: fluxModel,
          aspect_ratio: aspectRatio,
          characters: characters,
          scenes: scenes.map((s, idx) => ({
            id: s.id || idx + 1,
            text: s.text || '',
            duration: s.duration || '5s',
            characters: characters,
          })),
        }),
      });
      if (planRes.ok) {
        const planJson = await planRes.json();
        optimizedScenesData = (planJson.scenes || []).map((s, idx) => {
          const prev = prevScenes[idx];
          const sameAsPrev =
            prev &&
            prev.optimized_image_prompt === s.optimized_image_prompt &&
            prev.video_motion_prompt === s.video_motion_prompt;
          return enrichSceneWithVideoAndVO(s, idx, Boolean(sameAsPrev || !s.video_motion_prompt));
        });
      }
    } catch (err) {
      addLog(`[Card 2] Backend optimization notice: ${err.message}`, 'error');
    }

    if (optimizedScenesData.length === 0) {
      optimizedScenesData = scenes.map((s, idx) =>
        enrichSceneWithVideoAndVO(
          {
            id: s.id || idx + 1,
            scene_number: idx + 1,
            duration: s.duration || '5s',
            raw_prompt: s.text || '',
          },
          idx,
          true
        )
      );
    }

    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === '1') return { ...n, data: { ...n.data, isOptimizing: false, isActive: false } };
        if (n.id === '2') {
          return {
            ...n,
            data: {
              ...n.data,
              isOptimizing: false,
              isActive: false,
              fluxModel,
              aspectRatio,
              optimizedScenes: optimizedScenesData,
            },
          };
        }
        if (n.id === '4' || n.id === '5') {
          return {
            ...n,
            data: {
              ...n.data,
              isActive: false,
              aspectRatio,
              optimizedScenes: optimizedScenesData,
            },
          };
        }
        return n;
      })
    );

    if (optimizedScenesData.length > 0) {
      addLog(`[Card 2 & Card 4] Generated NEW ${fluxModel} image prompt(s) + NEW Video Motion & Voice-Over scripts!`, 'success');
      optimizedScenesData.forEach((s) => {
        addLog(`  ✨ [Scene ${s.scene_number} Image]: "${s.optimized_image_prompt.substring(0, 85)}..."`, 'info');
        addLog(`  🎙️ [Scene ${s.scene_number} VO (${s.voiceover_speaker})]: "${s.voiceover_english}"`, 'info');
      });
    }
    return optimizedScenesData;
  }, [setNodes, addLog]);

  optimizeOnlyRef.current = onOptimizeAllScenes;

  // Fresh Variation Prompt & Script for a single scene in Card 2 & Card 4
  const onRegenerateScene = useCallback(
    async (sceneIdx) => {
      const plannerNode = nodesRef.current.find((n) => n.id === '2');
      const prevScene = (plannerNode?.data?.optimizedScenes || [])[sceneIdx];

      setNodes((nds) =>
        nds.map((n) =>
          n.id === '2' || n.id === '4' ? { ...n, data: { ...n.data, isRegeneratingScene: sceneIdx } } : n
        )
      );

      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const scenes = promptNode?.data?.scenes || [];
      const targetRaw = scenes[sceneIdx] || { id: sceneIdx + 1, text: prevScene?.raw_prompt || '', duration: '5s' };
      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const fluxModel = promptNode?.data?.fluxModel || 'Qwen-Image-2.1';
      const characters = promptNode?.data?.characters || ['male', 'female'];

      addLog(`[Card 2 & Card 4] Generating NEW camera, lighting, motion & voice-over variation for Scene ${sceneIdx + 1}...`, 'info');

      let newScene = null;
      try {
        const res = await fetch('/api/v1/scenes/plan-multi', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            flux_model: fluxModel,
            aspect_ratio: aspectRatio,
            characters,
            scenes: [{ id: targetRaw.id || sceneIdx + 1, text: targetRaw.text, duration: targetRaw.duration || '5s', characters }],
          }),
        });
        if (res.ok) {
          const json = await res.json();
          if (json.scenes && json.scenes.length > 0) {
            const candidate = { ...json.scenes[0], scene_number: sceneIdx + 1 };
            const isSame =
              prevScene &&
              prevScene.optimized_image_prompt === candidate.optimized_image_prompt &&
              prevScene.video_motion_prompt === candidate.video_motion_prompt;
            newScene = enrichSceneWithVideoAndVO(candidate, sceneIdx, Boolean(isSame || !candidate.video_motion_prompt));
          }
        }
      } catch {}

      if (!newScene) {
        newScene = enrichSceneWithVideoAndVO(
          {
            ...(prevScene || {}),
            id: targetRaw.id || sceneIdx + 1,
            scene_number: sceneIdx + 1,
            raw_prompt: targetRaw.text || prevScene?.raw_prompt || '',
          },
          sceneIdx,
          true
        );
      }

      setNodes((nds) =>
        nds.map((n) => {
          if (n.id === '2') {
            const cur = [...(n.data.optimizedScenes || [])];
            cur[sceneIdx] = newScene;
            return { ...n, data: { ...n.data, optimizedScenes: cur, isRegeneratingScene: null } };
          }
          if (n.id === '4') {
            const cur = [...(n.data.optimizedScenes || [])];
            cur[sceneIdx] = newScene;
            return { ...n, data: { ...n.data, optimizedScenes: cur, isRegeneratingScene: null } };
          }
          return n;
        })
      );
      addLog(`[Card 2 & Card 4] Scene ${sceneIdx + 1} updated with NEW Master Prompt & NEW Motion/VO Script!`, 'success');
    },
    [setNodes, addLog]
  );

  // Regenerate a single image in Card 3 on Local GPU
  const onRegenerateFrame = useCallback(
    async (frameIdx) => {
      const keyframeNode = nodesRef.current.find((n) => n.id === '3');
      const plannerNode = nodesRef.current.find((n) => n.id === '2');
      const promptNode = nodesRef.current.find((n) => n.id === '1');

      const runId = keyframeNode?.data?.runId || `run_${Date.now()}_flux`;
      const sceneNum = frameIdx + 1;
      const optScene = (plannerNode?.data?.optimizedScenes || [])[frameIdx] || {};
      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const fluxModel = promptNode?.data?.fluxModel || 'FLUX.1-dev';

      setNodes((nds) =>
        nds.map((n) => (n.id === '3' ? { ...n, data: { ...n.data, isRegeneratingFrame: frameIdx, isActive: true } } : n))
      );

      addLog(`[Card 3: Local GPU] Synthesizing fresh master frame for Scene ${sceneNum} on RTX 4060...`, 'info');

      if (progressPollRef.current) clearInterval(progressPollRef.current);
      progressPollRef.current = setInterval(async () => {
        try {
          const pRes = await fetch('/api/v1/generate/progress');
          if (pRes.ok) {
            const pData = await pRes.json();
            if (pData && pData.active) {
              setLiveProgress(pData);
              setNodes((nds) =>
                nds.map((n) => (n.id === '3' ? { ...n, data: { ...n.data, liveProgress: pData } } : n))
              );
            }
          }
        } catch {}
      }, 500);

      try {
        const regenRes = await fetch('/api/v1/scenes/regenerate-keyframe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            run_id: runId,
            scene_number: sceneNum,
            prompt: optScene.optimized_image_prompt,
            raw_prompt: optScene.raw_prompt,
            camera: optScene.camera,
            lighting: optScene.lighting,
            aspect_ratio: aspectRatio,
            flux_model: fluxModel,
            steps: 28,
          }),
        });
        if (regenRes.ok) {
          const updatedFrame = await regenRes.json();
          const sep = updatedFrame.imageUrl.includes('?') ? '&' : '?';
          updatedFrame.imageUrl = `${updatedFrame.imageUrl}${sep}t=${Date.now()}`;

          setNodes((nds) =>
            nds.map((n) => {
              if (n.id === '3') {
                const cur = [...(n.data.frames || [])];
                cur[frameIdx] = updatedFrame;
                return {
                  ...n,
                  data: {
                    ...n.data,
                    frames: cur,
                    isRegeneratingFrame: null,
                    isActive: false,
                    liveProgress: null,
                  },
                };
              }
              if (n.id === '5') {
                const cur = [...(n.data.frames || [])];
                cur[frameIdx] = updatedFrame;
                return { ...n, data: { ...n.data, frames: cur } };
              }
              return n;
            })
          );
          addLog(`[Card 3: Local GPU] Scene ${sceneNum} image regenerated on RTX 4060 (Seed: ${updatedFrame.seed})!`, 'success');
        }
      } catch (err) {
        addLog(`[Card 3] Regeneration error: ${err.message}`, 'error');
        setNodes((nds) =>
          nds.map((n) => (n.id === '3' ? { ...n, data: { ...n.data, isRegeneratingFrame: null, isActive: false } } : n))
        );
      } finally {
        if (progressPollRef.current) {
          clearInterval(progressPollRef.current);
          progressPollRef.current = null;
        }
        setLiveProgress(null);
      }
    },
    [setNodes, addLog]
  );

  // Step 3: Synthesize Wan 2.2 Videos in Card 5 using Card 3 Frame Images + Card 4 Motion/VO Scripts
  const onGenerateAllVideos = useCallback(
    async (overrideRunId, overrideScenes, overrideFrames) => {
      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const keyframeNode = nodesRef.current.find((n) => n.id === '3');
      const scriptNode = nodesRef.current.find((n) => n.id === '4');
      const wanVideoNode = nodesRef.current.find((n) => n.id === '5');
      const plannerNode = nodesRef.current.find((n) => n.id === '2');

      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const wanModel = wanVideoNode?.data?.wanModel || 'Wan2.2-TI2V-5B';
      const runId = overrideRunId || keyframeNode?.data?.runId || wanVideoNode?.data?.runId || `run_${Date.now()}_flux`;
      const scenesToUse = overrideScenes || scriptNode?.data?.optimizedScenes || plannerNode?.data?.optimizedScenes || [];
      const framesToUse = overrideFrames || keyframeNode?.data?.frames || wanVideoNode?.data?.frames || [];

      if (scenesToUse.length === 0 || framesToUse.length === 0) {
        addLog('[Card 5] Waiting for both Card 3 (Frame Image) and Card 4 (Motion & VO Script) to be generated first.', 'error');
        return [];
      }

      setNodes((nds) =>
        nds.map((n) => (n.id === '5' ? { ...n, data: { ...n.data, isGeneratingVideos: true, isActive: true } } : n))
      );

      addLog(`[Card 3 + Card 4 ➔ Card 5] Synthesizing ${Math.max(scenesToUse.length, framesToUse.length)} Wan 2.2 Video(s) (${wanModel}) from Card 3 Frame Image + Card 4 Motion/VO Script...`, 'info');

      try {
        const vRes = await fetch('/api/v1/scenes/generate-videos', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            run_id: runId,
            wan_model: wanModel,
            aspect_ratio: aspectRatio,
            scenes: scenesToUse,
            frames: framesToUse,
          }),
        });

        if (vRes.ok) {
          const vJson = await vRes.json();
          const generatedVideos = vJson.videos || [];
          setNodes((nds) =>
            nds.map((n) =>
              n.id === '5'
                ? {
                    ...n,
                    data: {
                      ...n.data,
                      isGeneratingVideos: false,
                      isActive: false,
                      runId,
                      optimizedScenes: scenesToUse,
                      frames: framesToUse,
                      videos: generatedVideos,
                    },
                  }
                : n
            )
          );
          addLog(`=== [Card 5 COMPLETE] ${generatedVideos.length} Wan 2.2 Scene Video(s) + Voice-Over Subtitles Ready! ===`, 'success');
          return generatedVideos;
        }
      } catch (err) {
        addLog(`[Card 5 Error] ${err.message}`, 'error');
      }

      setNodes((nds) =>
        nds.map((n) => (n.id === '5' ? { ...n, data: { ...n.data, isGeneratingVideos: false, isActive: false } } : n))
      );
      return [];
    },
    [setNodes, addLog]
  );

  // Regenerate a single scene video in Card 5
  const onRegenerateSingleVideo = useCallback(
    async (sceneIdx) => {
      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const keyframeNode = nodesRef.current.find((n) => n.id === '3');
      const scriptNode = nodesRef.current.find((n) => n.id === '4');
      const wanVideoNode = nodesRef.current.find((n) => n.id === '5');
      const plannerNode = nodesRef.current.find((n) => n.id === '2');

      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const wanModel = wanVideoNode?.data?.wanModel || 'Wan2.2-TI2V-5B';
      const runId = keyframeNode?.data?.runId || wanVideoNode?.data?.runId || `run_${Date.now()}_flux`;
      const scenesList = scriptNode?.data?.optimizedScenes || plannerNode?.data?.optimizedScenes || [];
      const framesList = keyframeNode?.data?.frames || wanVideoNode?.data?.frames || [];

      const sc = scenesList[sceneIdx] || {};
      const fr = framesList[sceneIdx] || {};
      const sceneNum = sceneIdx + 1;

      setNodes((nds) =>
        nds.map((n) => (n.id === '5' ? { ...n, data: { ...n.data, isRegeneratingVideo: sceneIdx, isActive: true } } : n))
      );

      addLog(`[Card 5] Rendering Scene ${sceneNum} Wan 2.2 Video from Card 3 Frame Image + Card 4 Motion/VO Script...`, 'info');

      try {
        const res = await fetch('/api/v1/scenes/regenerate-video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            run_id: runId,
            scene_number: sceneNum,
            wan_model: wanModel,
            aspect_ratio: aspectRatio,
            source_image_url: fr.imageUrl,
            source_image_path: fr.file_path,
            video_motion_prompt: sc.video_motion_prompt || '',
            camera_motion: sc.camera_motion || 'Slow dolly push-in',
            voiceover_speaker: sc.voiceover_speaker || 'Arjun (Male Lead)',
            voiceover_sinhala: sc.voiceover_sinhala || '',
            voiceover_english: sc.voiceover_english || '',
            voice_tone: sc.voice_tone || 'Warm cinematic tone',
            sfx_cue: sc.sfx_cue || '',
          }),
        });

        if (res.ok) {
          const updatedVid = await res.json();
          setNodes((nds) =>
            nds.map((n) => {
              if (n.id === '5') {
                const cur = [...(n.data.videos || [])];
                cur[sceneIdx] = updatedVid;
                return {
                  ...n,
                  data: {
                    ...n.data,
                    videos: cur,
                    isRegeneratingVideo: null,
                    isActive: false,
                  },
                };
              }
              return n;
            })
          );
          addLog(`[Card 5] Scene ${sceneNum} Wan 2.2 video rendered successfully!`, 'success');
          return;
        }
      } catch (err) {
        addLog(`[Card 5] Video render error: ${err.message}`, 'error');
      }

      setNodes((nds) =>
        nds.map((n) => (n.id === '5' ? { ...n, data: { ...n.data, isRegeneratingVideo: null, isActive: false } } : n))
      );
    },
    [setNodes, addLog]
  );

  // Step 2: Generate Image(s) in Card 3 (Parallel to Card 4 Script) -> Then Feed Both into Card 5
  const generateImagesFromCard2 = useCallback(
    async (providedOptimizedScenes) => {
      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const plannerNode = nodesRef.current.find((n) => n.id === '2');

      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const fluxModel = promptNode?.data?.fluxModel || 'FLUX.1-dev';
      const scenes = promptNode?.data?.scenes || [{ id: 1, text: '', duration: '5s' }];
      const rawOptimizedScenes =
        providedOptimizedScenes || plannerNode?.data?.optimizedScenes || [];

      if (!rawOptimizedScenes || rawOptimizedScenes.length === 0) {
        addLog('[Card 2] Optimizing scenario first before GPU image generation...', 'info');
        const freshlyOptimized = await onOptimizeAllScenes();
        if (!freshlyOptimized || freshlyOptimized.length === 0) return;
        return generateImagesFromCard2(freshlyOptimized);
      }

      const optimizedScenesData = rawOptimizedScenes.map((sc, idx) => enrichSceneWithVideoAndVO(sc, idx));

      setIsRunning(true);
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id === '3') return { ...n, data: { ...n.data, isActive: true } };
          if (n.id === '4') return { ...n, data: { ...n.data, isActive: true, optimizedScenes: optimizedScenesData } };
          return { ...n, data: { ...n.data, isActive: false } };
        })
      );

      addLog(`[Parallel Step: Card 3 & Card 4] Synthesizing ${optimizedScenesData.length} Master Frame(s) in Card 3 + Generating Video Motion & Voice-Over Scripts in Card 4...`, 'info');

      if (progressPollRef.current) clearInterval(progressPollRef.current);
      let lastLoggedStep = -1;
      progressPollRef.current = setInterval(async () => {
        try {
          const pRes = await fetch('/api/v1/generate/progress');
          if (pRes.ok) {
            const pData = await pRes.json();
            if (pData && pData.active) {
              setLiveProgress(pData);
              setNodes((nds) =>
                nds.map((n) =>
                  n.id === '2' || n.id === '3' || n.id === '5'
                    ? { ...n, data: { ...n.data, liveProgress: pData } }
                    : n
                )
              );
              if (pData.node === 'card3' && pData.step !== lastLoggedStep && (pData.step === 1 || pData.step % 5 === 0 || pData.step === pData.total_steps)) {
                lastLoggedStep = pData.step;
                addLog(`[Card 3: RTX 4060 GPU] ${pData.message}`, pData.step === pData.total_steps ? 'success' : 'info');
              }
            }
          }
        } catch {}
      }, 500);

      try {
        const kfRes = await fetch('/api/v1/scenes/generate-keyframes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            flux_model: fluxModel,
            aspect_ratio: aspectRatio,
            steps: 28,
            scenes: optimizedScenesData,
          }),
        });

        if (!kfRes.ok) {
          throw new Error(`GPU Generation HTTP ${kfRes.status}`);
        }

        const keyframeRunData = await kfRes.json();
        const generatedKeyframes = (keyframeRunData.frames || []).map((f) => ({
          ...f,
          imageUrl: f.imageUrl ? `${f.imageUrl}${f.imageUrl.includes('?') ? '&' : '?'}t=${Date.now()}` : f.imageUrl,
        }));

        const runId = keyframeRunData.run_id;
        const folderPath = keyframeRunData.folder_path;

        setNodes((nds) =>
          nds.map((n) => {
            if (n.id === '3') {
              return {
                ...n,
                data: {
                  ...n.data,
                  isActive: false,
                  fluxModel,
                  runId,
                  folderPath,
                  frames: generatedKeyframes,
                },
              };
            }
            if (n.id === '4') {
              return {
                ...n,
                data: {
                  ...n.data,
                  isActive: false,
                  runId,
                  optimizedScenes: optimizedScenesData,
                },
              };
            }
            if (n.id === '5') {
              return {
                ...n,
                data: {
                  ...n.data,
                  runId,
                  optimizedScenes: optimizedScenesData,
                  frames: generatedKeyframes,
                },
              };
            }
            return { ...n, data: { ...n.data, isActive: false } };
          })
        );

        addLog(`=== [Card 3 & Card 4 COMPLETE] ${generatedKeyframes.length} Master Image(s) + Parallel Video Motion & Voice-Over Script(s) Ready! ===`, 'success');

        // Save to History
        try {
          await fetch('/api/v1/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              run_id: runId,
              title: `${fluxModel} (${aspectRatio} • ${scenes[0]?.text?.substring(0, 28) || 'Scene'}...)`,
              aspect_ratio: aspectRatio,
              flux_model: fluxModel,
              folder_path: folderPath,
              created_at: new Date().toISOString(),
              storyboard: { scenes, aspect_ratio: aspectRatio },
              planned_scenes: optimizedScenesData,
              keyframes: generatedKeyframes,
            }),
          });
          fetchHistoryList();
          setSelectedRunId(runId);
        } catch {}
      } catch (err) {
        // Check if backend GPU is still running or just completed a long Qwen-Image-2.1 run
        try {
          addLog(`[Card 3] Connection wait exceeded (${err.message}). Checking if GPU is still completing in background...`, 'info');
          let stillActive = true;
          for (let waitIdx = 0; waitIdx < 180; waitIdx++) {
            const pRes = await fetch('/api/v1/generate/progress');
            if (pRes.ok) {
              const pData = await pRes.json();
              if (!pData || !pData.active) {
                stillActive = false;
                break;
              }
            }
            await new Promise((r) => setTimeout(r, 2000));
          }
          if (!stillActive) {
            const hRes = await fetch('/api/v1/history');
            if (hRes.ok) {
              const hList = await hRes.json();
              if (Array.isArray(hList) && hList.length > 0) {
                setHistoryRuns(hList);
                await loadHistoryRun(hList[0].run_id, false);
                addLog(`=== [Card 3 & Card 4 COMPLETE] Recovered completed ${fluxModel} Master Frame (${hList[0].run_id})! ===`, 'success');
                return;
              }
            }
          }
        } catch {}
        addLog(`[Card 3 Error] ${err.message}`, 'error');
        setNodes((nds) => nds.map((n) => ({ ...n, data: { ...n.data, isActive: false } })));
      } finally {
        if (progressPollRef.current) {
          clearInterval(progressPollRef.current);
          progressPollRef.current = null;
        }
        setLiveProgress(null);
        setNodes((nds) =>
          nds.map((n) =>
            n.id === '2' || n.id === '3' || n.id === '5' ? { ...n, data: { ...n.data, liveProgress: null } } : n
          )
        );
        setIsRunning(false);
      }
    },
    [setNodes, addLog, onOptimizeAllScenes, onGenerateAllVideos, fetchHistoryList]
  );

  generateFromOptimizedRef.current = () => generateImagesFromCard2();

  // Full End-to-End Flow: Card 1 Scenario -> Card 2 Optimize -> (Card 3 Image || Card 4 Script) -> Card 5 Wan 2.2 Video
  const runWorkflow = async () => {
    if (isRunning) return;
    setIsRunning(true);
    const optimized = await onOptimizeAllScenes();
    if (optimized && optimized.length > 0) {
      await generateImagesFromCard2(optimized);
    } else {
      setIsRunning(false);
    }
  };

  runWorkflowRef.current = runWorkflow;

  // Bind callbacks to Nodes
  useEffect(() => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === '0') {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onSelectFluxModel: (m) => {
                setNodes((all) =>
                  all.map((n) => ({ ...n, data: { ...n.data, fluxModel: m } }))
                );
                addLog(`[Card 0] Selected Image Generation Model: ${m}`, 'info');
              },
              onSelectRatio: (val) => {
                setNodes((all) =>
                  all.map((n) => (n.id === '0' || n.id === '1' || n.id === '2' || n.id === '4' || n.id === '5' ? { ...n, data: { ...n.data, aspectRatio: val } } : n))
                );
                addLog(`Aspect ratio set: ${val}`, 'info');
              },
            },
          };
        }
        if (node.id === '1') {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onNewGeneration: startNewGeneration,
              onRunWorkflow: () => runWorkflowRef.current && runWorkflowRef.current(),
              onOptimizeOnly: () => optimizeOnlyRef.current && optimizeOnlyRef.current(),
              onSelectFluxModel: (m) => {
                setNodes((all) =>
                  all.map((n) => ({ ...n, data: { ...n.data, fluxModel: m } }))
                );
                addLog(`Selected model engine: ${m}`, 'info');
              },
              onSelectRatio: (val) => {
                setNodes((all) =>
                  all.map((n) => (n.id === '0' || n.id === '1' || n.id === '2' || n.id === '4' || n.id === '5' ? { ...n, data: { ...n.data, aspectRatio: val } } : n))
                );
                addLog(`Aspect ratio set: ${val}`, 'info');
              },
              onToggleCharacter: (charKey) => {
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '1') {
                      const cur = n.data.characters || ['male', 'female'];
                      const exists = cur.includes(charKey);
                      const next = exists ? cur.filter((c) => c !== charKey) : [...cur, charKey];
                      return { ...n, data: { ...n.data, characters: next } };
                    }
                    return n;
                  })
                );
              },
              onSetSceneCount: (cnt) => {
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '1') {
                      const cur = n.data.scenes || [{ id: 1, text: '', duration: '5s' }];
                      let updated = [...cur];
                      if (cnt > cur.length) {
                        for (let i = cur.length + 1; i <= cnt; i++) {
                          updated.push({ id: i, text: '', duration: '5s' });
                        }
                      } else if (cnt < cur.length) {
                        updated = updated.slice(0, cnt);
                      }
                      return { ...n, data: { ...n.data, scenes: updated } };
                    }
                    return n;
                  })
                );
                addLog(`Configured ${cnt} scenario scene(s).`, 'info');
              },
              onChangeSceneText: (idx, val) => {
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '1') {
                      const cur = [...(n.data.scenes || [{ id: 1, text: '', duration: '5s' }])];
                      if (cur[idx]) cur[idx] = { ...cur[idx], text: val };
                      return { ...n, data: { ...n.data, scenes: cur } };
                    }
                    return n;
                  })
                );
              },
            },
          };
        }
        if (node.id === '2') {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onOptimizePrompts: onOptimizeAllScenes,
              onRegenerateScene,
              onGenerateFromOptimized: () => generateFromOptimizedRef.current && generateFromOptimizedRef.current(),
              onEditOptimizedPrompt: (idx, newText) => {
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '2') {
                      const cur = [...(n.data.optimizedScenes || [])];
                      if (cur[idx]) {
                        cur[idx] = { ...cur[idx], optimized_image_prompt: newText };
                      }
                      return { ...n, data: { ...n.data, optimizedScenes: cur } };
                    }
                    return n;
                  })
                );
              },
            },
          };
        }
        if (node.id === '3') {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onRegenerateFrame,
            },
          };
        }
        if (node.id === '4') {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onRegenerateScene,
              onGenerateAllVideos: () => onGenerateAllVideos(),
              onEditVideoScriptField: (sceneIdx, fieldName, val) => {
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '2' || n.id === '4' || n.id === '5') {
                      const cur = [...(n.data.optimizedScenes || [])];
                      if (cur[sceneIdx]) {
                        cur[sceneIdx] = { ...cur[sceneIdx], [fieldName]: val };
                      }
                      return { ...n, data: { ...n.data, optimizedScenes: cur } };
                    }
                    return n;
                  })
                );
              },
            },
          };
        }
        if (node.id === '5') {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onGenerateAllVideos: () => onGenerateAllVideos(),
              onRegenerateSingleVideo,
              onSelectWanModel: (wm) => {
                setNodes((all) =>
                  all.map((n) => (n.id === '5' ? { ...n, data: { ...n.data, wanModel: wm } } : n))
                );
                addLog(`[Card 5] Selected Wan 2.2 Video Engine: ${wm}`, 'info');
              },
            },
          };
        }
        return node;
      })
    );
  }, [isRunning, setNodes, addLog, onOptimizeAllScenes, onRegenerateScene, onRegenerateFrame, onGenerateAllVideos, onRegenerateSingleVideo]);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      {/* Top Floating Studio Header */}
      <div className="studio-floating-header">
        <div className="studio-logo">
          <Sparkles size={18} style={{ color: '#a855f7' }} />
          <span>FILM AGENT • FLUX STUDIO</span>
        </div>

        {!isHeaderCollapsed && (
          <>
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                padding: '0.35rem 0.65rem',
                background: 'rgba(16, 185, 129, 0.12)',
                border: '1px solid rgba(16, 185, 129, 0.35)',
                borderRadius: '8px',
                fontSize: '0.72rem',
                color: '#34d399',
                fontWeight: 600,
              }}
            >
              <Cpu size={13} />
              <span>{gpuInfo}</span>
            </div>

            <button
              className="studio-header-btn primary"
              onClick={runWorkflow}
              disabled={isRunning}
            >
              {isRunning ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} fill="currentColor" />}
              <span>{isRunning ? 'Running GPU Pipeline...' : 'Optimize & Generate Image'}</span>
            </button>

            <button
              type="button"
              className="studio-header-btn"
              style={{
                background: 'rgba(56, 189, 248, 0.14)',
                borderColor: 'rgba(56, 189, 248, 0.45)',
                color: '#7dd3fc',
              }}
              onClick={startNewGeneration}
              disabled={isRunning}
              title="Clear loaded history and start a fresh New Generation"
            >
              <PlusCircle size={14} />
              <span>New Generation</span>
            </button>

            <div className="history-dropdown-wrapper" ref={historyRef}>
              <button
                type="button"
                className="studio-header-btn"
                onClick={() => setIsHistoryOpen((prev) => !prev)}
              >
                <History size={14} />
                <span>History</span>
                <span className="history-count-badge">{historyRuns.length}</span>
                <ChevronDown size={12} />
              </button>

              {isHistoryOpen && (
                <div className="history-menu">
                  <div className="history-menu-header">
                    <span>Saved Runs ({historyRuns.length})</span>
                    <button
                      type="button"
                      onClick={startNewGeneration}
                      style={{
                        background: 'rgba(56, 189, 248, 0.16)',
                        border: '1px solid rgba(56, 189, 248, 0.45)',
                        color: '#7dd3fc',
                        borderRadius: '5px',
                        padding: '2px 7px',
                        fontSize: '0.65rem',
                        fontWeight: 700,
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '3px',
                      }}
                    >
                      <PlusCircle size={11} /> New Generation
                    </button>
                  </div>
                  <div className="history-menu-list">
                    {historyRuns.length === 0 ? (
                      <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.74rem' }}>
                        <Clock size={18} style={{ margin: '0 auto 4px', opacity: 0.5 }} />
                        <div>No runs yet. Generate an image to save history!</div>
                      </div>
                    ) : (
                      historyRuns.map((run) => (
                        <div
                          key={run.run_id}
                          className="history-item"
                          onClick={() => loadHistoryRun(run.run_id)}
                        >
                          {run.thumbnail_url && (
                            <img src={run.thumbnail_url} alt={run.run_id} className="history-item-thumb" />
                          )}
                          <div className="history-item-details">
                            <div className="history-item-title">{run.title || run.run_id}</div>
                            <div className="history-item-sub">
                              <span className="history-item-pill">{run.aspect_ratio || '16:9'}</span>
                              <span>{run.scenes_count || 1} scene(s)</span>
                            </div>
                          </div>
                          {selectedRunId === run.run_id && <Check size={14} color="#34d399" />}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            <button className="studio-header-btn" onClick={resetLayout} title="Reset Node Positions">
              <RotateCcw size={14} />
              <span>Reset Layout</span>
            </button>

            <a href="/docs" target="_blank" rel="noreferrer" className="studio-header-btn">
              <ExternalLink size={14} />
              <span>API Docs</span>
            </a>
          </>
        )}

        <button
          className="studio-header-btn"
          onClick={() => setIsHeaderCollapsed((prev) => !prev)}
          style={{ padding: '0.45rem' }}
        >
          {isHeaderCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
        </button>
      </div>

      {/* React Flow Node Canvas */}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeDragStop={onNodeDragStop}
        nodeTypes={nodeTypes}
        fitView={false}
        defaultViewport={{ x: 25, y: 20, zoom: 0.78 }}
        minZoom={0.25}
        maxZoom={1.8}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1.8} color="#687182" />
        <Controls position="bottom-left" />
        <MiniMap
          position="bottom-left"
          width={200}
          height={114}
          nodeColor={(node) => {
            if (node.type === 'modelSelectorNode') return '#38bdf8';
            if (node.type === 'promptNode') return '#a855f7';
            if (node.type === 'plannerNode') return '#06b6d4';
            if (node.type === 'keyframeNode') return '#10b981';
            if (node.type === 'videoScriptNode') return '#f59e0b';
            return '#ffffff';
          }}
          maskColor="rgba(24, 28, 36, 0.85)"
        />
      </ReactFlow>

      {/* Bottom Right Execution Console */}
      <div className={`studio-floating-console ${isConsoleMinimized ? 'minimized' : ''}`}>
        <div className="console-header" onClick={() => setIsConsoleMinimized((prev) => !prev)}>
          <div className="console-header-title">
            <span className="status-indicator"></span>
            <Terminal size={14} />
            <span>FLUX Local GPU Execution Console</span>
          </div>
          <button type="button" className="toggle-console-btn">
            {isConsoleMinimized ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>

        {liveProgress && liveProgress.active && (
          <div className="console-live-progress">
            <div className="console-live-progress-header">
              <span className="console-live-progress-title">
                <Loader2 size={13} className="animate-spin" />
                {liveProgress.message}
              </span>
              <span className="console-live-progress-pct" style={{ color: '#34d399' }}>
                {liveProgress.percentage}%
              </span>
            </div>
            <div className="console-live-progress-bar">
              <div
                className="console-live-progress-fill"
                style={{
                  width: `${liveProgress.percentage}%`,
                  background: 'linear-gradient(90deg, #10b981, #06b6d4)',
                }}
              />
            </div>
          </div>
        )}

        <div className="console-logs-container">
          {logs.map((log) => (
            <div key={log.id} className={`log-entry ${log.type}`}>
              <span style={{ color: 'var(--text-dim)', marginRight: '6px' }}>[{log.time}]</span>
              {log.text}
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
}

