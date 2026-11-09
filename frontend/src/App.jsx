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
  NodeSceneStudio,
  NodeSceneVideo,
  enrichSceneWithVideoAndVO,
  segmentScriptIntoBeats,
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
  PlusCircle,
} from 'lucide-react';

const nodeTypes = {
  modelSelectorNode: NodeModelSelector,
  promptNode: NodePrompt,
  plannerNode: NodePlanner,
  sceneStudioNode: NodeSceneStudio,
  sceneVideoNode: NodeSceneVideo,
};

const SCENE_EDGE_COLORS = ['#10b981', '#f59e0b', '#38bdf8', '#a855f7', '#ec4899', '#06b6d4'];
const VIDEO_EDGE_COLORS = ['#f43f5e', '#ec4899', '#f59e0b', '#38bdf8', '#a855f7', '#10b981'];

function getSceneCardPosition(sceneIdx, totalScenes) {
  if (totalScenes === 1) {
    return { x: 1385, y: 110 };
  }
  const rowsPerCol = totalScenes <= 3 ? 3 : totalScenes <= 12 ? 4 : 5;
  const col = Math.floor(sceneIdx / rowsPerCol);
  const row = sceneIdx % rowsPerCol;
  return {
    x: 1385 + col * 990,
    y: 20 + row * 690,
  };
}

function getVideoCardPosition(sceneIdx, totalScenes) {
  const scPos = getSceneCardPosition(sceneIdx, totalScenes);
  return {
    x: scPos.x + 485,
    y: scPos.y,
  };
}

function buildSceneStudioNode(
  sceneIdx,
  totalScenes,
  fluxModel = 'Qwen-Image-2.1',
  aspectRatio = '16:9',
  existingNode = null,
  wanModel = 'LTX-Video-2B'
) {
  const id = `scene-${sceneIdx + 1}`;
  return {
    id,
    type: 'sceneStudioNode',
    position: existingNode?.position || getSceneCardPosition(sceneIdx, totalScenes),
    data: {
      ...(existingNode?.data || {}),
      sceneIndex: sceneIdx,
      sceneNumber: sceneIdx + 1,
      fluxModel: existingNode?.data?.fluxModel || fluxModel,
      wanModel: existingNode?.data?.wanModel || wanModel,
      aspectRatio: existingNode?.data?.aspectRatio || aspectRatio,
      frame: existingNode?.data?.frame || null,
      sceneScript: existingNode?.data?.sceneScript || null,
    },
  };
}

function buildSceneVideoNode(
  sceneIdx,
  totalScenes,
  wanModel = 'LTX-Video-2B',
  aspectRatio = '16:9',
  existingNode = null
) {
  const id = `video-${sceneIdx + 1}`;
  return {
    id,
    type: 'sceneVideoNode',
    position: existingNode?.position || getVideoCardPosition(sceneIdx, totalScenes),
    data: {
      ...(existingNode?.data || {}),
      sceneIndex: sceneIdx,
      sceneNumber: sceneIdx + 1,
      wanModel: existingNode?.data?.wanModel || wanModel,
      aspectRatio: existingNode?.data?.aspectRatio || aspectRatio,
      frame: existingNode?.data?.frame || null,
      sceneScript: existingNode?.data?.sceneScript || null,
      videoClip: existingNode?.data?.videoClip || null,
    },
  };
}

function buildEdgesForSceneCount(sceneCount) {
  const baseEdges = [
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
  ];

  for (let i = 0; i < sceneCount; i++) {
    const color = SCENE_EDGE_COLORS[i % SCENE_EDGE_COLORS.length];
    const vidColor = VIDEO_EDGE_COLORS[i % VIDEO_EDGE_COLORS.length];
    baseEdges.push({
      id: `e2-scene-${i + 1}`,
      source: '2',
      target: `scene-${i + 1}`,
      animated: true,
      style: {
        stroke: color,
        strokeWidth: 2.6,
        filter: `drop-shadow(0 0 6px ${color}b0)`,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color },
    });
    baseEdges.push({
      id: `e-scene-${i + 1}-video-${i + 1}`,
      source: `scene-${i + 1}`,
      target: `video-${i + 1}`,
      animated: true,
      style: {
        stroke: vidColor,
        strokeWidth: 2.6,
        filter: `drop-shadow(0 0 6px ${vidColor}b0)`,
      },
      markerEnd: { type: MarkerType.ArrowClosed, color: vidColor },
    });
  }
  return baseEdges;
}

const DEFAULT_STORY_SCRIPT =
  "the way you threat your mother is the way life will threat you. listen carefully, audience , this is a lesson many people learn only when is too late. a mother's love is one of the few gifts in this world that ask for nothing in return. before you ever spoke your first word, she was already speaking prayers over your life , before you took first step, she was carrying your weight without complaint. she sacrificed her comfort so that you could have a better future. never take that love for granted.";

const DEFAULT_NODES = [
  {
    id: '0',
    type: 'modelSelectorNode',
    position: { x: 20, y: 210 },
    data: {
      fluxModel: 'Qwen-Image-2.1',
      wanModel: 'LTX-Video-2B',
      aspectRatio: '16:9',
    },
  },
  {
    id: '1',
    type: 'promptNode',
    position: { x: 455, y: 210 },
    data: {
      fluxModel: 'Qwen-Image-2.1',
      wanModel: 'LTX-Video-2B',
      aspectRatio: '16:9',
      characters: ['female'],
      extractedSceneCount: 0,
      storyText: DEFAULT_STORY_SCRIPT,
      scenes: [
        {
          id: 1,
          text: DEFAULT_STORY_SCRIPT,
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
      wanModel: 'LTX-Video-2B',
      aspectRatio: '16:9',
      storyTitle: '',
      optimizedScenes: [],
    },
  },
  buildSceneStudioNode(0, 1, 'Qwen-Image-2.1', '16:9', null, 'LTX-Video-2B'),
  buildSceneVideoNode(0, 1, 'LTX-Video-2B', '16:9'),
];

const DEFAULT_EDGES = buildEdgesForSceneCount(1);

const STORAGE_KEY = 'film_agent_dynamic_scene_nodes_v9';

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState(DEFAULT_NODES);
  const [edges, setEdges, onEdgesChange] = useEdgesState(DEFAULT_EDGES);

  const [isRunning, setIsRunning] = useState(false);
  const [isHeaderCollapsed, setIsHeaderCollapsed] = useState(false);
  const [isConsoleMinimized, setIsConsoleMinimized] = useState(false);
  const [gpuInfo, setGpuInfo] = useState('RTX 4060 (8GB VRAM)');
  const [logs, setLogs] = useState([
    { id: 1, time: new Date().toLocaleTimeString(), text: '[System] FILM AGENT — Qwen-Image-2.1 & FLUX Studio initialized.', type: 'info' },
    { id: 2, time: new Date().toLocaleTimeString(), text: '[Card 1] Full Story Input & Character Turnaround Reference Sheets ready.', type: 'success' },
    { id: 3, time: new Date().toLocaleTimeString(), text: '[Card 2] Gemini 3 Flash AI Story Breakdown & Prompt Director ready.', type: 'success' },
    { id: 4, time: new Date().toLocaleTimeString(), text: '[Scene Studio] Dynamic per-scene Frame Image + Motion & VO Script cards active.', type: 'info' },
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

  // Fetch History List & GPU status on mount (Does NOT auto-load old history into cards!)
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

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const selectedCharactersRef = useRef(['female']);

  const resetLayout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    const sceneNodesCount = nodesRef.current.filter((n) => n.id.startsWith('scene-')).length || 1;
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === '0') return { ...n, position: { x: 20, y: 210 } };
        if (n.id === '1') return { ...n, position: { x: 455, y: 210 } };
        if (n.id === '2') return { ...n, position: { x: 910, y: 210 } };
        if (n.id.startsWith('scene-')) {
          const idx = parseInt(n.id.replace('scene-', ''), 10) - 1;
          return { ...n, position: getSceneCardPosition(idx, sceneNodesCount) };
        }
        if (n.id.startsWith('video-')) {
          const idx = parseInt(n.id.replace('video-', ''), 10) - 1;
          return { ...n, position: getVideoCardPosition(idx, sceneNodesCount) };
        }
        return n;
      })
    );
    setEdges(buildEdgesForSceneCount(sceneNodesCount));
    addLog('Canvas layout reset to default 3.K ➔ 4.K storyboard grid.', 'info');
  }, [setNodes, setEdges, addLog]);

  const onConnect = useCallback((params) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  // Start a Fresh New Story Generation (Clears loaded History Run & resets to 1 clean Scene & Video Card)
  const startNewGeneration = useCallback(() => {
    setSelectedRunId(null);
    setIsHistoryOpen(false);
    setNodes((nds) => {
      const modelNode = nds.find((n) => n.id === '0');
      const currentModel = modelNode?.data?.fluxModel || 'Qwen-Image-2.1';
      const currentWanModel = modelNode?.data?.wanModel || 'LTX-Video-2B';
      const currentRatio = modelNode?.data?.aspectRatio || '16:9';

      return [
        {
          id: '0',
          type: 'modelSelectorNode',
          position: nds.find((n) => n.id === '0')?.position || { x: 20, y: 210 },
          data: {
            fluxModel: currentModel,
            wanModel: currentWanModel,
            aspectRatio: currentRatio,
          },
        },
        {
          id: '1',
          type: 'promptNode',
          position: nds.find((n) => n.id === '1')?.position || { x: 455, y: 210 },
          data: {
            fluxModel: currentModel,
            wanModel: currentWanModel,
            aspectRatio: currentRatio,
            characters: nds.find((n) => n.id === '1')?.data?.characters || ['female'],
            storyText: DEFAULT_STORY_SCRIPT,
            extractedSceneCount: 0,
            scenes: [{ id: 1, text: DEFAULT_STORY_SCRIPT, duration: '5s' }],
            isActive: false,
            isOptimizing: false,
          },
        },
        {
          id: '2',
          type: 'plannerNode',
          position: nds.find((n) => n.id === '2')?.position || { x: 910, y: 210 },
          data: {
            fluxModel: currentModel,
            wanModel: currentWanModel,
            aspectRatio: currentRatio,
            storyTitle: '',
            optimizedScenes: [],
            isActive: false,
            isOptimizing: false,
          },
        },
        buildSceneStudioNode(0, 1, currentModel, currentRatio, null, currentWanModel),
        buildSceneVideoNode(0, 1, currentWanModel, currentRatio),
      ];
    });
    setEdges(buildEdgesForSceneCount(1));
    addLog('[New Generation] Cleared previous state. Select speaker & enter script in Card 1!', 'success');
  }, [setNodes, setEdges, addLog]);

  // Restore a History Run ONLY when the user clicks a History item in the History dropdown
  const loadHistoryRun = useCallback(
    async (runId, keepCurrentModel = false) => {
      try {
        addLog(`[History] Restoring run: ${runId}...`, 'info');
        const res = await fetch(`/api/v1/history/${runId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        setSelectedRunId(runId);
        setIsHistoryOpen(false);

        const restoredScenes =
          data.storyboard?.scenes?.length > 0
            ? data.storyboard.scenes
            : (data.planned_scenes || []).map((ps, idx) => ({
                id: idx + 1,
                text: ps.raw_prompt || '',
                duration: ps.duration || '5s',
              }));
        const restoredStoryText =
          data.storyboard?.story_text ||
          restoredScenes
            .map((s) => s.text)
            .filter(Boolean)
            .join('\n\n');
        const enrichedPlanned = (data.planned_scenes || []).map((ps, idx) => enrichSceneWithVideoAndVO(ps, idx, false));
        const restoredFrames = (data.keyframes || []).map((f) => ({
          ...f,
          imageUrl: f.imageUrl || f.url,
        }));
        const restoredVideos = data.videos || [];
        const count = Math.max(restoredScenes.length, enrichedPlanned.length, restoredFrames.length, 1);

        setNodes((nds) => {
          const modelNode = nds.find((n) => n.id === '0');
          const targetModel = keepCurrentModel
            ? modelNode?.data?.fluxModel || 'Qwen-Image-2.1'
            : data.flux_model || modelNode?.data?.fluxModel || 'Qwen-Image-2.1';
          const targetWanModel = modelNode?.data?.wanModel || 'LTX-Video-2B';
          const targetRatio = data.aspect_ratio || modelNode?.data?.aspectRatio || '16:9';

          const baseNodes = [
            {
              id: '0',
              type: 'modelSelectorNode',
              position: nds.find((n) => n.id === '0')?.position || { x: 20, y: 210 },
              data: {
                ...(nds.find((n) => n.id === '0')?.data || {}),
                fluxModel: targetModel,
                wanModel: targetWanModel,
                aspectRatio: targetRatio,
              },
            },
            {
              id: '1',
              type: 'promptNode',
              position: nds.find((n) => n.id === '1')?.position || { x: 455, y: 210 },
              data: {
                ...(nds.find((n) => n.id === '1')?.data || {}),
                fluxModel: targetModel,
                wanModel: targetWanModel,
                aspectRatio: targetRatio,
                storyText: restoredStoryText,
                extractedSceneCount: count,
                scenes: restoredScenes.length > 0 ? restoredScenes : [{ id: 1, text: '', duration: '5s' }],
              },
            },
            {
              id: '2',
              type: 'plannerNode',
              position: nds.find((n) => n.id === '2')?.position || { x: 910, y: 210 },
              data: {
                ...(nds.find((n) => n.id === '2')?.data || {}),
                fluxModel: targetModel,
                wanModel: targetWanModel,
                aspectRatio: targetRatio,
                storyTitle: data.title || '',
                optimizedScenes: enrichedPlanned,
              },
            },
          ];

          for (let i = 0; i < count; i++) {
            const existing = nds.find((n) => n.id === `scene-${i + 1}`);
            const scNode = buildSceneStudioNode(i, count, targetModel, targetRatio, existing, targetWanModel);
            scNode.position = getSceneCardPosition(i, count);
            scNode.data.runId = data.run_id;
            scNode.data.folderPath = data.folder_path || `data/keyframes/${data.run_id}`;
            scNode.data.frame = restoredFrames[i] || restoredFrames[0] || null;
            scNode.data.sceneScript = enrichedPlanned[i] || null;
            scNode.data.isActive = false;
            baseNodes.push(scNode);

            const existingVid = nds.find((n) => n.id === `video-${i + 1}`);
            const vidNode = buildSceneVideoNode(i, count, targetWanModel, targetRatio, existingVid);
            vidNode.position = getVideoCardPosition(i, count);
            vidNode.data.runId = data.run_id;
            vidNode.data.frame = restoredFrames[i] || restoredFrames[0] || null;
            vidNode.data.sceneScript = enrichedPlanned[i] || null;
            vidNode.data.videoClip = restoredVideos[i] || null;
            vidNode.data.isActive = false;
            baseNodes.push(vidNode);
          }
          return baseNodes;
        });

        setEdges(buildEdgesForSceneCount(count));
        addLog(`[History] Restored "${data.title || runId}" into ${count} Scene (3.K) & Video (4.K) Card(s).`, 'success');
      } catch (err) {
        addLog(`[History] Failed to restore run: ${err.message}`, 'error');
      }
    },
    [addLog, setNodes, setEdges]
  );

  // Step 1: AI Automatic Full Story Breakdown (Card 1 -> Card 2 -> Spawns N Dynamic 3.K Scene & 4.K Video Cards)
  const onOptimizeAllScenes = useCallback(async () => {
    const modelNode = nodesRef.current.find((n) => n.id === '0');
    const promptNode = nodesRef.current.find((n) => n.id === '1');
    const plannerNode = nodesRef.current.find((n) => n.id === '2');
    const prevScenes = plannerNode?.data?.optimizedScenes || [];

    const fallbackScenes = promptNode?.data?.scenes || [{ id: 1, text: '', duration: '5s' }];
    const storyText = (
      promptNode?.data?.storyText !== undefined
        ? promptNode.data.storyText
        : fallbackScenes.map((s) => s.text).filter(Boolean).join('\n\n')
    ).trim();

    const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
    const fluxModel = promptNode?.data?.fluxModel || 'Qwen-Image-2.1';
    const wanModel = modelNode?.data?.wanModel || 'LTX-Video-2B';
    const characters = selectedCharactersRef.current || promptNode?.data?.characters || ['female'];

    setSelectedRunId(null);

    setNodes((nds) =>
      nds.map((n) => {
        if (n.id === '1') return { ...n, data: { ...n.data, isOptimizing: true, isActive: true } };
        if (n.id === '2') return { ...n, data: { ...n.data, isOptimizing: true, isActive: true, optimizedScenes: [] } };
        return n;
      })
    );

    addLog(`[Card 1 ➔ Card 2] Analyzing Spoken Script & Designing Fresh Studio Master Frame via Gemini 3 Flash AI...`, 'info');

    const optPoll = setInterval(async () => {
      try {
        const pRes = await fetch('/api/v1/generate/progress');
        if (pRes.ok) {
          const pData = await pRes.json();
          if (pData && pData.active && pData.node === 'card2') {
            setLiveProgress(pData);
            setNodes((nds) =>
              nds.map((n) => (n.id === '2' ? { ...n, data: { ...n.data, liveProgress: pData } } : n))
            );
          }
        }
      } catch {}
    }, 220);

    let optimizedScenesData = [];
    let storyTitle = '';
    try {
      const planRes = await fetch('/api/v1/scenes/breakdown-story', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          story_text: storyText || 'Success never comes overnight; it is built in the quiet hours when nobody is watching.',
          flux_model: fluxModel,
          aspect_ratio: aspectRatio,
          characters: characters,
        }),
      });
      if (planRes.ok) {
        const planJson = await planRes.json();
        storyTitle = planJson.story_title || '';
        optimizedScenesData = (planJson.scenes || []).map((s, idx) => {
          const prev = prevScenes[idx];
          const sameAsPrev =
            prev &&
            prev.optimized_image_prompt === s.optimized_image_prompt &&
            prev.video_motion_prompt === s.video_motion_prompt;
          return enrichSceneWithVideoAndVO(s, idx, Boolean(sameAsPrev || !s.video_motion_prompt), characters);
        });
      }
    } catch (err) {
      addLog(`[Card 2] Backend story breakdown notice: ${err.message}`, 'error');
    } finally {
      clearInterval(optPoll);
      setLiveProgress(null);
    }

    if (optimizedScenesData.length === 0) {
      const rawBeats = segmentScriptIntoBeats(
        storyText || 'Success never comes overnight; it is built in the quiet hours when nobody is watching.'
      );
      const beatsToUse = rawBeats.length > 0 ? rawBeats : [storyText || 'Spoken script segment'];
      optimizedScenesData = beatsToUse.map((beatText, idx) =>
        enrichSceneWithVideoAndVO(
          {
            id: idx + 1,
            scene_number: idx + 1,
            duration: '5s',
            raw_prompt: beatText,
          },
          idx,
          true,
          characters
        )
      );
    }

    const extractedCount = Math.max(optimizedScenesData.length, 1);
    const extractedSceneItems = optimizedScenesData.map((s, idx) => ({
      id: idx + 1,
      text: s.raw_prompt || '',
      duration: s.duration || '5s',
    }));

    setEdges(buildEdgesForSceneCount(extractedCount));

    setNodes((nds) => {
      const coreNodes = nds
        .filter((n) => n.id === '0' || n.id === '1' || n.id === '2')
        .map((n) => {
          if (n.id === '1') {
            return {
              ...n,
              data: {
                ...n.data,
                isOptimizing: false,
                isActive: false,
                extractedSceneCount: extractedCount,
                scenes: extractedSceneItems,
              },
            };
          }
          if (n.id === '2') {
            return {
              ...n,
              data: {
                ...n.data,
                isOptimizing: false,
                isActive: false,
                liveProgress: null,
                fluxModel,
                wanModel,
                aspectRatio,
                storyTitle,
                optimizedScenes: optimizedScenesData,
              },
            };
          }
          return n;
        });

      const dynamicNodes = [];
      for (let i = 0; i < extractedCount; i++) {
        const existing = nds.find((n) => n.id === `scene-${i + 1}`);
        const scNode = buildSceneStudioNode(i, extractedCount, fluxModel, aspectRatio, existing, wanModel);
        scNode.position = getSceneCardPosition(i, extractedCount);
        scNode.data.isActive = false;
        scNode.data.fluxModel = fluxModel;
        scNode.data.wanModel = wanModel;
        scNode.data.aspectRatio = aspectRatio;
        scNode.data.sceneScript = optimizedScenesData[i] || null;
        dynamicNodes.push(scNode);

        const existingVid = nds.find((n) => n.id === `video-${i + 1}`);
        const vidNode = buildSceneVideoNode(i, extractedCount, wanModel, aspectRatio, existingVid);
        vidNode.position = getVideoCardPosition(i, extractedCount);
        vidNode.data.isActive = false;
        vidNode.data.wanModel = wanModel;
        vidNode.data.aspectRatio = aspectRatio;
        vidNode.data.sceneScript = optimizedScenesData[i] || null;
        vidNode.data.videoClip = null;
        dynamicNodes.push(vidNode);
      }
      return [...coreNodes, ...dynamicNodes];
    });

    if (optimizedScenesData.length > 0) {
      addLog(
        `[Card 2 ➔ 3.1..3.${extractedCount} ➔ 4.1..4.${extractedCount}] Created 1 Fresh Master Frame Prompt & broke script into ${extractedCount} Scene Cards (3.K) + ${extractedCount} Video Cards (4.K)!`,
        'success'
      );
      optimizedScenesData.forEach((s) => {
        addLog(`  🎬 [3.${s.scene_number} ➔ 4.${s.scene_number} Script]: "${s.raw_prompt}"`, 'info');
      });
    }
    return optimizedScenesData;
  }, [setNodes, setEdges, addLog]);

  optimizeOnlyRef.current = onOptimizeAllScenes;

  // Fresh Variation Prompt & Script for a single scene in Card 2 & Scene N Studio Card
  const onRegenerateScene = useCallback(
    async (sceneIdx) => {
      const plannerNode = nodesRef.current.find((n) => n.id === '2');
      const prevScene = (plannerNode?.data?.optimizedScenes || [])[sceneIdx];

      setNodes((nds) =>
        nds.map((n) =>
          n.id === '2' || n.id === `scene-${sceneIdx + 1}`
            ? { ...n, data: { ...n.data, isRegeneratingScene: sceneIdx } }
            : n
        )
      );

      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const scenes = promptNode?.data?.scenes || [];
      const targetRaw = scenes[sceneIdx] || { id: sceneIdx + 1, text: prevScene?.raw_prompt || '', duration: '5s' };
      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const fluxModel = promptNode?.data?.fluxModel || 'Qwen-Image-2.1';
      const characters = selectedCharactersRef.current || promptNode?.data?.characters || ['female'];

      addLog(`[Scene ${sceneIdx + 1} Studio] Generating NEW camera, lighting, motion & voice-over variation for Scene ${sceneIdx + 1}...`, 'info');

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
            newScene = enrichSceneWithVideoAndVO(candidate, sceneIdx, Boolean(isSame || !candidate.video_motion_prompt), characters);
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
          true,
          characters
        );
      }

      setNodes((nds) =>
        nds.map((n) => {
          if (n.id === '2') {
            const cur = [...(n.data.optimizedScenes || [])];
            cur[sceneIdx] = newScene;
            return { ...n, data: { ...n.data, optimizedScenes: cur, isRegeneratingScene: null } };
          }
          if (n.id === `scene-${sceneIdx + 1}` || n.id === `video-${sceneIdx + 1}`) {
            return { ...n, data: { ...n.data, sceneScript: newScene, isRegeneratingScene: null } };
          }
          return n;
        })
      );
      addLog(`[Scene ${sceneIdx + 1} Studio] Updated Scene ${sceneIdx + 1} (3.${sceneIdx + 1} & 4.${sceneIdx + 1}) with NEW Master Prompt & NEW Motion/VO Script!`, 'success');
    },
    [setNodes, addLog]
  );

  // Regenerate a single image in Scene N Studio Card on Local GPU
  const onRegenerateFrame = useCallback(
    async (frameIdx) => {
      const targetSceneNode = nodesRef.current.find((n) => n.id === `scene-${frameIdx + 1}`);
      const plannerNode = nodesRef.current.find((n) => n.id === '2');
      const promptNode = nodesRef.current.find((n) => n.id === '1');

      const runId = targetSceneNode?.data?.runId || `run_${Date.now()}_flux`;
      const sceneNum = frameIdx + 1;
      const optScene =
        targetSceneNode?.data?.sceneScript ||
        (plannerNode?.data?.optimizedScenes || [])[frameIdx] ||
        {};
      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const fluxModel = promptNode?.data?.fluxModel || 'Qwen-Image-2.1';
      const characters = selectedCharactersRef.current || promptNode?.data?.characters || ['female'];
      const wantsMale = characters.includes('male') && !characters.includes('female');

      setNodes((nds) =>
        nds.map((n) =>
          n.id === `scene-${sceneNum}`
            ? { ...n, data: { ...n.data, isRegeneratingFrame: frameIdx, isActive: true } }
            : n
        )
      );

      addLog(`[Scene ${sceneNum} Studio: Local GPU] Synthesizing fresh master frame for Scene ${sceneNum} on RTX 4060...`, 'info');

      if (progressPollRef.current) clearInterval(progressPollRef.current);
      progressPollRef.current = setInterval(async () => {
        try {
          const pRes = await fetch('/api/v1/generate/progress');
          if (pRes.ok) {
            const pData = await pRes.json();
            if (pData && pData.active) {
              setLiveProgress(pData);
              setNodes((nds) =>
                nds.map((n) =>
                  n.id === `scene-${sceneNum}` ? { ...n, data: { ...n.data, liveProgress: pData } } : n
                )
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
            characters,
            voiceover_speaker: wantsMale ? 'Arjun (Male Lead)' : 'Maya (Female Lead)',
            steps: 28,
          }),
        });
        if (regenRes.ok) {
          const updatedFrame = await regenRes.json();
          const sep = updatedFrame.imageUrl.includes('?') ? '&' : '?';
          updatedFrame.imageUrl = `${updatedFrame.imageUrl}${sep}t=${Date.now()}`;

          setNodes((nds) =>
            nds.map((n) => {
              if (n.id.startsWith('scene-') || n.id.startsWith('video-')) {
                return {
                  ...n,
                  data: {
                    ...n.data,
                    frame: updatedFrame,
                    isRegeneratingFrame: null,
                    isActive: false,
                    liveProgress: null,
                  },
                };
              }
              return n;
            })
          );
          addLog(`[Master Frame Regenerated] Brand-new Studio Master Frame synthesized on RTX 4060 (Seed: ${updatedFrame.seed}) and synced across all 3.K & 4.K cards!`, 'success');
        }
      } catch (err) {
        addLog(`[Scene ${sceneNum} Studio] Regeneration error: ${err.message}`, 'error');
        setNodes((nds) =>
          nds.map((n) =>
            n.id === `scene-${sceneNum}`
              ? { ...n, data: { ...n.data, isRegeneratingFrame: null, isActive: false } }
              : n
          )
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

  // Generate/Regenerate a single 4.K Scene Video from 3.K Master Frame + Script (3.K ➔ 4.K)
  const onGenerateSceneVideo = useCallback(
    async (sceneIdx) => {
      const sceneNum = sceneIdx + 1;
      const modelNode = nodesRef.current.find((n) => n.id === '0');
      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const plannerNode = nodesRef.current.find((n) => n.id === '2');
      const sceneNode = nodesRef.current.find((n) => n.id === `scene-${sceneNum}`);
      const videoNode = nodesRef.current.find((n) => n.id === `video-${sceneNum}`);

      const wanModel = videoNode?.data?.wanModel || modelNode?.data?.wanModel || 'LTX-Video-2B';
      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const runId = sceneNode?.data?.runId || videoNode?.data?.runId || `run_${Date.now()}_flux`;
      const frame = sceneNode?.data?.frame || videoNode?.data?.frame || null;
      const sceneScript =
        sceneNode?.data?.sceneScript ||
        videoNode?.data?.sceneScript ||
        (plannerNode?.data?.optimizedScenes || [])[sceneIdx] ||
        {};

      if (!frame || (!frame.imageUrl && !frame.filePath)) {
        addLog(`[4.${sceneNum} Video Studio] Master Frame not generated yet in 3.${sceneNum}! Please generate the Master Frame first.`, 'error');
        return;
      }

      setNodes((nds) =>
        nds.map((n) =>
          n.id === `video-${sceneNum}` || n.id === `scene-${sceneNum}`
            ? { ...n, data: { ...n.data, isGeneratingVideos: true, isRegeneratingVideo: sceneIdx, isActive: n.id === `video-${sceneNum}` } }
            : n
        )
      );

      addLog(`[3.${sceneNum} ➔ 4.${sceneNum} Video Studio] Synthesizing Scene ${sceneNum} Video Clip with ${wanModel}...`, 'info');

      if (progressPollRef.current) clearInterval(progressPollRef.current);
      progressPollRef.current = setInterval(async () => {
        try {
          const pRes = await fetch('/api/v1/generate/progress');
          if (pRes.ok) {
            const pData = await pRes.json();
            if (pData && pData.active) {
              setLiveProgress(pData);
              setNodes((nds) =>
                nds.map((n) =>
                  n.id === `video-${sceneNum}` ? { ...n, data: { ...n.data, liveProgress: pData } } : n
                )
              );
            }
          }
        } catch {}
      }, 500);

      try {
        const res = await fetch('/api/v1/scenes/regenerate-video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            run_id: runId,
            scene_number: sceneNum,
            wan_model: wanModel,
            aspect_ratio: aspectRatio,
            source_image_url: frame.imageUrl || null,
            source_image_path: frame.filePath || null,
            video_motion_prompt: sceneScript.video_motion_prompt || sceneScript.raw_prompt || '',
            camera_motion: sceneScript.camera_motion || sceneScript.camera || '',
            voiceover_speaker: sceneScript.voiceover_speaker || 'Narrator',
            voiceover_sinhala: sceneScript.voiceover_sinhala || '',
            voiceover_english: sceneScript.voiceover_english || '',
            voice_tone: sceneScript.voice_tone || '',
            sfx_cue: sceneScript.sfx_cue || '',
          }),
        });

        if (!res.ok) {
          throw new Error(`Video Generation HTTP ${res.status}`);
        }

        const clip = await res.json();
        const sep = clip.videoUrl && clip.videoUrl.includes('?') ? '&' : '?';
        if (clip.videoUrl) {
          clip.videoUrl = `${clip.videoUrl}${sep}t=${Date.now()}`;
        }

        setNodes((nds) =>
          nds.map((n) => {
            if (n.id === `video-${sceneNum}`) {
              return {
                ...n,
                data: {
                  ...n.data,
                  videoClip: clip,
                  frame,
                  sceneScript,
                  runId,
                  wanModel,
                  isGeneratingVideos: false,
                  isRegeneratingVideo: null,
                  isActive: false,
                  liveProgress: null,
                },
              };
            }
            if (n.id === `scene-${sceneNum}`) {
              return {
                ...n,
                data: {
                  ...n.data,
                  isGeneratingVideos: false,
                  isRegeneratingVideo: null,
                },
              };
            }
            return n;
          })
        );

        addLog(`[3.${sceneNum} ➔ 4.${sceneNum} Complete] Scene ${sceneNum} Video (${wanModel}) ready!`, 'success');
      } catch (err) {
        addLog(`[4.${sceneNum} Video Error] ${err.message}`, 'error');
        setNodes((nds) =>
          nds.map((n) =>
            n.id === `video-${sceneNum}` || n.id === `scene-${sceneNum}`
              ? { ...n, data: { ...n.data, isGeneratingVideos: false, isRegeneratingVideo: null, isActive: false, liveProgress: null } }
              : n
          )
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

  // Batch Generate All 4.1 .. 4.N Scene Videos from 3.1 .. 3.N Cards (3.1 ➔ 4.1, 3.2 ➔ 4.2, ...)
  const onGenerateAllVideos = useCallback(
    async (overrideRunId, overrideScenes, overrideFrames) => {
      const modelNode = nodesRef.current.find((n) => n.id === '0');
      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const plannerNode = nodesRef.current.find((n) => n.id === '2');

      const wanModel = modelNode?.data?.wanModel || 'LTX-Video-2B';
      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';

      const sceneNodes = nodesRef.current
        .filter((n) => n.id.startsWith('scene-'))
        .sort((a, b) => (a.data.sceneIndex ?? 0) - (b.data.sceneIndex ?? 0));

      const runId =
        overrideRunId ||
        sceneNodes[0]?.data?.runId ||
        `run_${Date.now()}_flux`;

      const scenesPayload =
        overrideScenes ||
        sceneNodes.map((sn, idx) => sn.data?.sceneScript || (plannerNode?.data?.optimizedScenes || [])[idx] || {});

      const framesPayload =
        overrideFrames ||
        sceneNodes
          .map((sn) => sn.data?.frame)
          .filter(Boolean);

      if (!framesPayload || framesPayload.length === 0) {
        addLog('[4.K Video Studio] Please generate the Master Frame in Card 3.K first before generating videos.', 'error');
        return;
      }

      addLog(`[3.1..3.${scenesPayload.length} ➔ 4.1..4.${scenesPayload.length}] Synthesizing ${scenesPayload.length} Scene Video Clip(s) one-by-one using ${wanModel}...`, 'info');

      if (progressPollRef.current) clearInterval(progressPollRef.current);
      progressPollRef.current = setInterval(async () => {
        try {
          const pRes = await fetch('/api/v1/generate/progress');
          if (pRes.ok) {
            const pData = await pRes.json();
            if (pData && pData.active) {
              setLiveProgress(pData);
              setNodes((nds) =>
                nds.map((n) =>
                  n.id.startsWith('video-') ? { ...n, data: { ...n.data, liveProgress: pData } } : n
                )
              );
            }
          }
        } catch {}
      }, 500);

      const videos = [];
      try {
        for (let idx = 0; idx < scenesPayload.length; idx++) {
          const sceneNum = idx + 1;
          const sc = scenesPayload[idx] || {};
          const fr = framesPayload[idx] || framesPayload[0] || {};

          // Highlight ONLY the currently rendering 4.K video card so 3.1..3.N images remain crisp & visible
          setNodes((nds) =>
            nds.map((n) => {
              if (n.id === `video-${sceneNum}`) {
                return {
                  ...n,
                  data: {
                    ...n.data,
                    isGeneratingVideos: true,
                    isRegeneratingVideo: idx,
                    isActive: true,
                  },
                };
              }
              if (n.id.startsWith('video-')) {
                return {
                  ...n,
                  data: {
                    ...n.data,
                    isGeneratingVideos: true,
                    isRegeneratingVideo: idx,
                    isActive: false,
                  },
                };
              }
              if (n.id.startsWith('scene-')) {
                return {
                  ...n,
                  data: {
                    ...n.data,
                    isGeneratingVideos: true,
                    isRegeneratingVideo: idx,
                    isActive: false,
                  },
                };
              }
              return n;
            })
          );

          addLog(`[3.${sceneNum} ➔ 4.${sceneNum}] Synthesizing Scene ${sceneNum}/${scenesPayload.length} Video (${wanModel})...`, 'info');

          const res = await fetch('/api/v1/scenes/regenerate-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              run_id: runId,
              scene_number: sceneNum,
              wan_model: wanModel,
              aspect_ratio: aspectRatio,
              source_image_url: fr.imageUrl || fr.url,
              source_image_path: fr.file_path || fr.filePath,
              video_motion_prompt: sc.video_motion_prompt || '',
              camera_motion: sc.camera_motion || 'Slow dolly push-in',
              voiceover_speaker: sc.voiceover_speaker || 'Maya (Female Lead)',
              voiceover_sinhala: sc.voiceover_sinhala || '',
              voiceover_english: sc.voiceover_english || sc.raw_prompt || '',
              voice_tone: sc.voice_tone || 'Warm cinematic tone',
              sfx_cue: sc.sfx_cue || 'Natural cinema ambience',
            }),
          });

          if (!res.ok) {
            throw new Error(`Scene ${sceneNum} Video Generation HTTP ${res.status}`);
          }

          const singleVideo = await res.json();
          const sep = singleVideo.videoUrl && singleVideo.videoUrl.includes('?') ? '&' : '?';
          if (singleVideo.videoUrl) {
            singleVideo.videoUrl = `${singleVideo.videoUrl}${sep}t=${Date.now()}`;
          }
          videos.push(singleVideo);

          // Immediately display this completed video in Card 4.{sceneNum}
          setNodes((nds) =>
            nds.map((n) => {
              if (n.id === `video-${sceneNum}`) {
                return {
                  ...n,
                  data: {
                    ...n.data,
                    isActive: false,
                    liveProgress: null,
                    wanModel,
                    runId,
                    videoClip: singleVideo,
                  },
                };
              }
              return n;
            })
          );

          addLog(`✓ [4.${sceneNum} Ready] Scene ${sceneNum} Video Clip rendered and displayed!`, 'success');
        }

        setNodes((nds) =>
          nds.map((n) =>
            n.id.startsWith('video-') || n.id.startsWith('scene-')
              ? {
                  ...n,
                  data: {
                    ...n.data,
                    isActive: false,
                    isGeneratingVideos: false,
                    isRegeneratingVideo: null,
                    liveProgress: null,
                  },
                }
              : n
          )
        );

        addLog(`=== [VIDEO PIPELINE COMPLETE] Generated ${videos.length} Scene Video Clip(s) (3.1 ➔ 4.1 .. 3.${videos.length} ➔ 4.${videos.length}) with ${wanModel}! ===`, 'success');
        return videos;
      } catch (err) {
        addLog(`[Batch Video Generation Error] ${err.message}`, 'error');
        setNodes((nds) =>
          nds.map((n) =>
            n.id.startsWith('video-') || n.id.startsWith('scene-')
              ? { ...n, data: { ...n.data, isActive: false, isGeneratingVideos: false, isRegeneratingVideo: null, liveProgress: null } }
              : n
          )
        );
        return videos;
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

  // Step 2: Generate Master Frame Image(s) into each Scene N Studio Card (3.K) and then Video Cards (4.K)
  const generateImagesFromCard2 = useCallback(
    async (providedOptimizedScenes) => {
      const modelNode = nodesRef.current.find((n) => n.id === '0');
      const promptNode = nodesRef.current.find((n) => n.id === '1');
      const plannerNode = nodesRef.current.find((n) => n.id === '2');

      const aspectRatio = promptNode?.data?.aspectRatio || '16:9';
      const fluxModel = promptNode?.data?.fluxModel || 'Qwen-Image-2.1';
      const characters = selectedCharactersRef.current || promptNode?.data?.characters || ['female'];
      const wantsMale = characters.includes('male') && !characters.includes('female');
      const wanModel = modelNode?.data?.wanModel || 'LTX-Video-2B';
      const scenes = promptNode?.data?.scenes || [{ id: 1, text: '', duration: '5s' }];
      const rawOptimizedScenes =
        providedOptimizedScenes || plannerNode?.data?.optimizedScenes || [];

      const firstSceneText = (
        (rawOptimizedScenes[0]?.voiceover_speaker || '') +
        ' ' +
        (rawOptimizedScenes[0]?.character_description || '') +
        ' ' +
        (rawOptimizedScenes[0]?.optimized_image_prompt || '')
      ).toLowerCase();
      const isGenderMismatch =
        rawOptimizedScenes.length > 0 &&
        ((wantsMale && (firstSceneText.includes('maya') || !firstSceneText.includes('arjun'))) ||
          (!wantsMale && firstSceneText.includes('arjun') && !firstSceneText.includes('maya')));

      if (!rawOptimizedScenes || rawOptimizedScenes.length === 0 || isGenderMismatch) {
        addLog('[Card 2] Optimizing scenario for selected Speaker Character before GPU image generation...', 'info');
        const freshlyOptimized = await onOptimizeAllScenes();
        if (!freshlyOptimized || freshlyOptimized.length === 0) return;
        return generateImagesFromCard2(freshlyOptimized);
      }

      const optimizedScenesData = rawOptimizedScenes.map((sc, idx) =>
        enrichSceneWithVideoAndVO(sc, idx, false, characters)
      );
      const sceneCount = Math.max(optimizedScenesData.length, 1);

      setIsRunning(true);
      setEdges(buildEdgesForSceneCount(sceneCount));
      setNodes((nds) => {
        const coreNodes = nds
          .filter((n) => n.id === '0' || n.id === '1' || n.id === '2')
          .map((n) => ({ ...n, data: { ...n.data, isActive: false } }));

        const sceneNodes = [];
        const videoNodes = [];
        for (let i = 0; i < sceneCount; i++) {
          const existing = nds.find((n) => n.id === `scene-${i + 1}`);
          const scNode = buildSceneStudioNode(i, sceneCount, fluxModel, aspectRatio, existing);
          scNode.data.isActive = true;
          scNode.data.frame = null;
          scNode.data.runId = null;
          scNode.data.sceneScript = optimizedScenesData[i] || null;
          sceneNodes.push(scNode);

          const existingVid = nds.find((n) => n.id === `video-${i + 1}`);
          const vidNode = buildSceneVideoNode(i, sceneCount, wanModel, aspectRatio, existingVid);
          vidNode.data.isActive = false;
          vidNode.data.frame = null;
          vidNode.data.videoClip = null;
          vidNode.data.runId = null;
          vidNode.data.sceneScript = optimizedScenesData[i] || null;
          videoNodes.push(vidNode);
        }
        return [...coreNodes, ...sceneNodes, ...videoNodes];
      });

      addLog(`[Card 2 ➔ ${sceneCount} Scene Studio Card(s)] Synthesizing Master Frame on Local RTX 4060 GPU...`, 'info');

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
                  n.id === '2' || n.id.startsWith('scene-')
                    ? { ...n, data: { ...n.data, liveProgress: pData } }
                    : n
                )
              );
              if (pData.node === 'card3' && pData.step !== lastLoggedStep && (pData.step === 1 || pData.step % 4 === 0 || pData.step === pData.total_steps)) {
                lastLoggedStep = pData.step;
                addLog(`[RTX 4060 GPU] ${pData.message}`, pData.step === pData.total_steps ? 'success' : 'info');
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
            if (n.id.startsWith('scene-')) {
              const idx = n.data.sceneIndex ?? parseInt(n.id.replace('scene-', ''), 10) - 1;
              return {
                ...n,
                data: {
                  ...n.data,
                  isActive: false,
                  liveProgress: null,
                  fluxModel,
                  runId,
                  folderPath,
                  frame: generatedKeyframes[idx] || generatedKeyframes[0] || null,
                  sceneScript: optimizedScenesData[idx] || n.data.sceneScript || null,
                },
              };
            }
            if (n.id.startsWith('video-')) {
              const idx = n.data.sceneIndex ?? parseInt(n.id.replace('video-', ''), 10) - 1;
              return {
                ...n,
                data: {
                  ...n.data,
                  isActive: false,
                  liveProgress: null,
                  wanModel,
                  runId,
                  folderPath,
                  frame: generatedKeyframes[idx] || generatedKeyframes[0] || null,
                  sceneScript: optimizedScenesData[idx] || n.data.sceneScript || null,
                },
              };
            }
            return { ...n, data: { ...n.data, isActive: false, liveProgress: null } };
          })
        );

        addLog(`=== [3.1..3.${sceneCount} READY] Master Frame + Scripts populated! Starting 3.K ➔ 4.K Video Synthesis (${wanModel})... ===`, 'success');

        // Ensure browser loads & paints the generated Master Frame into all 3.1..3.N cards before starting GPU video synthesis
        if (generatedKeyframes.length > 0 && generatedKeyframes[0]?.imageUrl) {
          await new Promise((resolve) => {
            const preloadImg = new window.Image();
            const timer = setTimeout(resolve, 900);
            preloadImg.onload = () => {
              clearTimeout(timer);
              resolve();
            };
            preloadImg.onerror = () => {
              clearTimeout(timer);
              resolve();
            };
            preloadImg.src = generatedKeyframes[0].imageUrl;
          });
        }
        await new Promise((resolve) => setTimeout(resolve, 350));

        // Automatically synthesize 3.1 ➔ 4.1, 3.2 ➔ 4.2, ..., 3.N ➔ 4.N videos one-by-one
        const generatedVideos = await onGenerateAllVideos(runId, optimizedScenesData, generatedKeyframes);

        // Save new run to History (for future browsing only)
        try {
          const storyTextForHistory =
            promptNode?.data?.storyText ||
            optimizedScenesData
              .map((s) => s.raw_prompt)
              .filter(Boolean)
              .join('\n\n');
          const storyTitleForHistory = plannerNode?.data?.storyTitle || '';
          await fetch('/api/v1/history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              run_id: runId,
              title:
                storyTitleForHistory ||
                `${fluxModel} + ${wanModel} (${aspectRatio} • ${optimizedScenesData.length} Scenes • ${(storyTextForHistory || 'Story').substring(0, 26)}...)`,
              aspect_ratio: aspectRatio,
              flux_model: fluxModel,
              wan_model: wanModel,
              folder_path: folderPath,
              created_at: new Date().toISOString(),
              storyboard: {
                story_text: storyTextForHistory,
                scenes: optimizedScenesData.map((s, idx) => ({
                  id: idx + 1,
                  text: s.raw_prompt || '',
                  duration: s.duration || '5s',
                })),
                aspect_ratio: aspectRatio,
              },
              planned_scenes: optimizedScenesData,
              keyframes: generatedKeyframes,
              videos: generatedVideos || [],
            }),
          });
          fetchHistoryList();
          setSelectedRunId(runId);
        } catch {}
      } catch (err) {
        // NEVER load old history on error! Report the error cleanly.
        addLog(`[GPU Generation Error] ${err.message}`, 'error');
        setNodes((nds) => nds.map((n) => ({ ...n, data: { ...n.data, isActive: false, liveProgress: null } })));
      } finally {
        if (progressPollRef.current) {
          clearInterval(progressPollRef.current);
          progressPollRef.current = null;
        }
        setLiveProgress(null);
        setNodes((nds) =>
          nds.map((n) =>
            n.id === '2' || n.id.startsWith('scene-') || n.id.startsWith('video-')
              ? { ...n, data: { ...n.data, liveProgress: null } }
              : n
          )
        );
        setIsRunning(false);
      }
    },
    [setNodes, setEdges, addLog, onOptimizeAllScenes, onGenerateAllVideos, fetchHistoryList]
  );

  generateFromOptimizedRef.current = () => generateImagesFromCard2();

  // Full End-to-End Flow: Card 1 Scenario -> Card 2 Optimize -> Dynamic Scene Studio Cards (3.K) -> Video Studio Cards (4.K)
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
              onSelectWanModel: (vm) => {
                setNodes((all) =>
                  all.map((n) => ({ ...n, data: { ...n.data, wanModel: vm } }))
                );
                addLog(`[Card 0] Selected Video Generation Model: ${vm}`, 'info');
              },
              onSelectRatio: (val) => {
                setNodes((all) =>
                  all.map((n) => ({ ...n, data: { ...n.data, aspectRatio: val } }))
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
                  all.map((n) => ({ ...n, data: { ...n.data, aspectRatio: val } }))
                );
                addLog(`Aspect ratio set: ${val}`, 'info');
              },
              onToggleCharacter: (charKey) => {
                selectedCharactersRef.current = [charKey];
                setSelectedRunId(null);
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '1') {
                      const liveCount = segmentScriptIntoBeats(n.data.storyText || '').length;
                      return { ...n, data: { ...n.data, characters: [charKey], extractedSceneCount: liveCount } };
                    }
                    if (n.id === '2') {
                      return { ...n, data: { ...n.data, optimizedScenes: [], storyTitle: '' } };
                    }
                    return n;
                  })
                );
                addLog(
                  `[Card 1] Selected Speaker Character: ${charKey === 'female' ? 'Girl (Maya — Female Lead)' : 'Man (Arjun — Male Lead)'}`,
                  'info'
                );
              },
              onSetSceneCount: (cnt) => {
                setSelectedRunId(null);
                setEdges(buildEdgesForSceneCount(cnt));
                setNodes((all) => {
                  const modelNode = all.find((n) => n.id === '0');
                  const fluxModel = modelNode?.data?.fluxModel || 'Qwen-Image-2.1';
                  const wanModel = modelNode?.data?.wanModel || 'LTX-Video-2B';
                  const aspectRatio = modelNode?.data?.aspectRatio || '16:9';

                  const coreNodes = all
                    .filter((n) => n.id === '0' || n.id === '1' || n.id === '2')
                    .map((n) => {
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
                    });

                  const sceneNodes = [];
                  const videoNodes = [];
                  for (let i = 0; i < cnt; i++) {
                    const existing = all.find((n) => n.id === `scene-${i + 1}`);
                    const scNode = buildSceneStudioNode(i, cnt, fluxModel, aspectRatio, existing);
                    scNode.position = getSceneCardPosition(i, cnt);
                    sceneNodes.push(scNode);

                    const existingVid = all.find((n) => n.id === `video-${i + 1}`);
                    const vidNode = buildSceneVideoNode(i, cnt, wanModel, aspectRatio, existingVid);
                    vidNode.position = getVideoCardPosition(i, cnt);
                    videoNodes.push(vidNode);
                  }
                  return [...coreNodes, ...sceneNodes, ...videoNodes];
                });
                addLog(`[Canvas] Spawned ${cnt} Scene Studio Card(s) (3.1..3.${cnt}) ➔ ${cnt} Video Studio Card(s) (4.1..4.${cnt}).`, 'info');
              },
              onChangeStoryText: (val) => {
                setSelectedRunId(null);
                const liveCount = segmentScriptIntoBeats(val || '').length;
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '1') {
                      return {
                        ...n,
                        data: {
                          ...n.data,
                          storyText: val,
                          extractedSceneCount: liveCount,
                        },
                      };
                    }
                    if (n.id === '2') {
                      return {
                        ...n,
                        data: {
                          ...n.data,
                          optimizedScenes: [],
                          storyTitle: '',
                        },
                      };
                    }
                    return n;
                  })
                );
              },
              onChangeSceneText: (idx, val) => {
                setSelectedRunId(null);
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
        if (node.id.startsWith('scene-')) {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onRegenerateFrame,
              onRegenerateScene,
              onGenerateSceneVideo,
              onEditVideoScriptField: (sceneIdx, fieldName, val) => {
                setNodes((all) =>
                  all.map((n) => {
                    if (n.id === '2') {
                      const cur = [...(n.data.optimizedScenes || [])];
                      if (cur[sceneIdx]) {
                        cur[sceneIdx] = { ...cur[sceneIdx], [fieldName]: val };
                      }
                      return { ...n, data: { ...n.data, optimizedScenes: cur } };
                    }
                    if (n.id === `scene-${sceneIdx + 1}` || n.id === `video-${sceneIdx + 1}`) {
                      const updatedScript = { ...(n.data.sceneScript || {}), [fieldName]: val };
                      return { ...n, data: { ...n.data, sceneScript: updatedScript } };
                    }
                    return n;
                  })
                );
              },
            },
          };
        }
        if (node.id.startsWith('video-')) {
          return {
            ...node,
            data: {
              ...node.data,
              isRunning,
              onGenerateSceneVideo,
              onGenerateAllVideos: () => onGenerateAllVideos(),
              onSelectWanModel: (vm) => {
                setNodes((all) =>
                  all.map((n) => ({ ...n, data: { ...n.data, wanModel: vm } }))
                );
                addLog(`[4.K Video Studio] Selected Video Generation Model: ${vm}`, 'info');
              },
            },
          };
        }
        return node;
      })
    );
  }, [isRunning, setNodes, setEdges, addLog, startNewGeneration, onOptimizeAllScenes, onRegenerateScene, onRegenerateFrame, onGenerateSceneVideo, onGenerateAllVideos]);

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
            if (node.type === 'sceneStudioNode') return '#10b981';
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
