"use client"

import { useState, useRef, useEffect } from "react"
import dynamic from "next/dynamic"
import { Download, Plus, RefreshCw, Send, Circle as Sphere, CuboidIcon as Cube, Square } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { ChatOpenAI } from "@langchain/openai";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
// 修改导入，使用别名区分库组件和自定义组件
import { default as CodeEditorComponent } from "@uiw/react-textarea-code-editor";
import type { MessageContent } from "@langchain/core/messages";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
const ShaderPreview = dynamic(() => import("@/components/shader-preview"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="animate-pulse text-primary">Loading preview...</div>
    </div>
  ),
})

class SimpleMemorySaver {
  private storage = new Map<string, any>();
  
  async save(config: { configurable: { thread_id: string } }, value: any) {
    this.storage.set(config.configurable.thread_id, value);
  }
  
  async load(config: { configurable: { thread_id: string } }) {
    return this.storage.get(config.configurable.thread_id);
  }
}
// Default shader code
const DEFAULT_VERTEX_SHADER = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`

const DEFAULT_FRAGMENT_SHADER = `
uniform float uTime;
varying vec2 vUv;

vec3 palette(float t) {
  vec3 a = vec3(0.5, 0.5, 0.5);
  vec3 b = vec3(0.5, 0.5, 0.5);
  vec3 c = vec3(1.0, 1.0, 1.0);
  vec3 d = vec3(0.263, 0.416, 0.557);
  return a + b * cos(6.28318 * (c * t + d));
}

void main() {
  vec2 uv = vUv;
  vec2 center = vec2(0.5);
  vec2 pos = uv - center;
  
  float radius = length(pos) * 2.0;
  float angle = atan(pos.y, pos.x);
  
  // Add some movement
  radius += sin(angle * 8.0 + uTime) * 0.1;
  
  vec3 color = palette(radius + uTime * 0.4);
  
  // Add some glow
  color += 0.1 / (radius * radius + 0.05);
  
  gl_FragColor = vec4(color, 1.0);
}
`

// 在文件顶部添加这个噪声函数库对象
const GLSL_NOISE_FUNCTIONS = {
  // Simplex噪声
  snoise: {
    declaration: /float\s+snoise\s*\(\s*vec2\s+[a-zA-Z0-9_]+\s*\)\s*;/,
    implementation: `
// Simplex 2D noise
float snoise(vec2 v) {
  // Precomputed values for skewed triangular grid
  const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  
  // First corner (x0)
  vec2 i  = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  
  // Other two corners (x1, x2)
  vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  
  // Permutations
  i = mod(i, 289.0);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  
  // Gradients
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m*m;
  m = m*m;
  
  // Gradients
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  
  // Normalise gradients
  m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
  
  // Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

vec3 permute(vec3 x) {
  return mod(((x*34.0)+1.0)*x, 289.0);
}`
  },
  
  // Perlin噪声
  pnoise: {
    declaration: /float\s+pnoise\s*\(\s*vec[23]\s+[a-zA-Z0-9_]+\s*(?:,\s*float\s+[a-zA-Z0-9_]+\s*)?\)\s*;/,
    implementation: `
// Perlin noise
float pnoise(vec2 p, float freq) {
  float unit = freq;
  vec2 ij = floor(p * unit);
  vec2 xy = fract(p * unit);
  
  // Quintic interpolation curve
  xy = xy * xy * xy * (xy * (xy * 6.0 - 15.0) + 10.0);
  
  // Four corners in 2D of a tile
  float a = random(ij);
  float b = random(ij + vec2(1.0, 0.0));
  float c = random(ij + vec2(0.0, 1.0));
  float d = random(ij + vec2(1.0, 1.0));
  
  // Mix the four corners
  float x1 = mix(a, b, xy.x);
  float x2 = mix(c, d, xy.x);
  return mix(x1, x2, xy.y);
}

float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}`
  },
  
  // Cellular/Worley噪声
  cellular: {
    declaration: /vec2\s+cellular\s*\(\s*vec2\s+[a-zA-Z0-9_]+\s*\)\s*;/,
    implementation: `
// Cellular noise (Worley)
vec2 cellular(vec2 P) {
  const float K = 0.142857142857; // 1/7
  const float Ko = 0.428571428571; // 3/7
  
  vec2 Pi = mod(floor(P), 289.0);
  vec2 Pf = fract(P);
  vec3 oi = vec3(-1.0, 0.0, 1.0);
  vec3 of = vec3(-0.5, 0.5, 1.5);
  vec3 px = permute(Pi.x + oi);
  vec3 p = permute(px.x + Pi.y + oi);
  vec3 ox = fract(p*K) - Ko;
  vec3 oy = mod(floor(p*K),7.0)*K - Ko;
  vec3 dx = Pf.x + 0.5 + ox;
  vec3 dy = Pf.y - of + oy;
  vec3 d1 = dx * dx + dy * dy; // squared distances
  p = permute(px.y + Pi.y + oi);
  ox = fract(p*K) - Ko;
  oy = mod(floor(p*K),7.0)*K - Ko;
  dx = Pf.x - 0.5 + ox;
  dy = Pf.y - of + oy;
  vec3 d2 = dx * dx + dy * dy;
  p = permute(px.z + Pi.y + oi);
  ox = fract(p*K) - Ko;
  oy = mod(floor(p*K),7.0)*K - Ko;
  dx = Pf.x - 1.5 + ox;
  dy = Pf.y - of + oy;
  vec3 d3 = dx * dx + dy * dy;
  
  // Sort out the two smallest distances
  vec3 d = min(min(d1,d2),d3);
  d.xy = (d.x < d.y) ? d.xy : d.yx; // Swap if needed
  d.xz = (d.x < d.z) ? d.xz : d.zx;
  d.yz = (d.y < d.z) ? d.yz : d.zy;
  
  return sqrt(d.xy);
}`
  },
  
  // FBM (Fractal Brownian Motion)
  fbm: {
    declaration: /float\s+fbm\s*\(\s*vec[23]\s+[a-zA-Z0-9_]+\s*\)\s*;/,
    implementation: `
// Fractal Brownian Motion
float fbm(vec2 p) {
  float f = 0.0;
  float w = 0.5;
  float freq = 1.0;
  for (int i = 0; i < 5; i++) {
    f += w * snoise(p * freq);
    freq *= 2.0;
    w *= 0.5;
  }
  return f;
}`
  },
  
  // 3D Simplex Noise
  snoise3: {
    declaration: /float\s+snoise\s*\(\s*vec3\s+[a-zA-Z0-9_]+\s*\)\s*;/,
    implementation: `
// 3D Simplex noise
float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  
  // First corner
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  
  // Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  
  // x1, x2, x3
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - 0.5;
  
  // Permutations
  i = mod(i, 289.0);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
           
  // Gradients (NxN points over a square, mapped onto an octahedron)
  float n_ = 1.0/7.0; // 1.0/7.0 = N^-1
  vec3 ns = n_ * D.wyz - D.xzx;
  
  vec4 j = p - 49.0 * floor(p * ns.z *ns.z);
  
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  
  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  
  // Normalize gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  
  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

vec4 permute(vec4 x) {
  return mod(((x*34.0)+1.0)*x, 289.0);
}

vec4 taylorInvSqrt(vec4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
`
  },
  
  // HSL to RGB转换
  hsl2rgb: {
    declaration: /vec3\s+hsl2rgb\s*\(\s*vec3\s+[a-zA-Z0-9_]+\s*\)\s*;/,
    implementation: `
// HSL to RGB conversion
vec3 hsl2rgb(vec3 hsl) {
  vec3 rgb = clamp(abs(mod(hsl.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return hsl.z + hsl.y * (rgb - 0.5) * (1.0 - abs(2.0 * hsl.z - 1.0));
}`
  },
  
  // RGB to HSL转换
  rgb2hsl: {
    declaration: /vec3\s+rgb2hsl\s*\(\s*vec3\s+[a-zA-Z0-9_]+\s*\)\s*;/,
    implementation: `
// RGB to HSL conversion
vec3 rgb2hsl(vec3 color) {
  float maxColor = max(max(color.r, color.g), color.b);
  float minColor = min(min(color.r, color.g), color.b);
  float delta = maxColor - minColor;
  
  float h = 0.0;
  float s = 0.0;
  float l = (maxColor + minColor) / 2.0;
  
  if (delta > 0.0) {
    s = (l < 0.5) ? (delta / (maxColor + minColor)) : (delta / (2.0 - maxColor - minColor));
    
    if (maxColor == color.r) {
      h = (color.g - color.b) / delta + (color.g < color.b ? 6.0 : 0.0);
    } else if (maxColor == color.g) {
      h = (color.b - color.r) / delta + 2.0;
    } else {
      h = (color.r - color.g) / delta + 4.0;
    }
    h /= 6.0;
  }
  
  return vec3(h, s, l);
}`
  }
};

// 添加依赖检测函数
const checkShaderDependencies = (shaderCode: string | null | undefined) => {
  if (!shaderCode) return shaderCode;
  
  // 预处理：如果着色器中引用了某些函数但没有声明
  let modifiedCode = shaderCode;
  let madeChanges = false;
  
  // 跟踪已添加的函数以避免重复
  const addedFunctions = new Set<string>();
  
  // 更精确地检测函数定义 - 更多的函数定义模式
  const hasFunctionDefinition = (funcName: string) => {
    // 检查常见的函数定义模式
    const patterns = [
      new RegExp(`${funcName}\\s*\\([^)]*\\)\\s*\\{`), // 函数实现：name() { ... }
      new RegExp(`${funcName}\\s*\\([^)]*\\);`),       // 函数声明：name();
      new RegExp(`vec[234]\\s+${funcName}\\s*\\(`),    // vec3 name(
      new RegExp(`float\\s+${funcName}\\s*\\(`),       // float name(
      new RegExp(`void\\s+${funcName}\\s*\\(`)         // void name(
    ];
    
    return patterns.some(pattern => pattern.test(modifiedCode));
  };
  
  // 安全地添加函数 - 避免重复
  const addFunction = (funcName: string, implementation: string) => {
    if (addedFunctions.has(funcName) || hasFunctionDefinition(funcName)) {
      return false; // 已经添加过或者已经定义了
    }
    
    modifiedCode = `${implementation}\n${modifiedCode}`;
    addedFunctions.add(funcName);
    madeChanges = true;
    return true;
  };

  // 核心工具函数 - 优先添加最基础的函数
  if (modifiedCode.includes("permute(") && !hasFunctionDefinition("permute")) {
    // 检查是否是vec3或vec4类型的permute
    if (modifiedCode.includes("permute(vec3")) {
      addFunction("permute_vec3", `
// Permutation function
vec3 permute(vec3 x) {
  return mod(((x*34.0)+1.0)*x, 289.0);
}`);
    } 
    
    if (modifiedCode.includes("permute(vec4")) {
      addFunction("permute_vec4", `
// Permutation function for vec4
vec4 permute(vec4 x) {
  return mod(((x*34.0)+1.0)*x, 289.0);
}`);
    }
  }

  if (modifiedCode.includes("random(") && !hasFunctionDefinition("random")) {
    addFunction("random", `
// Random hash function
float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}`);
  }
  
  // 检查并添加所有缺失的噪声函数，先检查是否调用，避免不必要添加
  for (const [name, func] of Object.entries(GLSL_NOISE_FUNCTIONS)) {
    // 检查是否调用了函数但没有定义
    const functionCallPattern = new RegExp(`\\b${name}\\s*\\(`);
    if (functionCallPattern.test(modifiedCode) && !hasFunctionDefinition(name)) {
      // 在着色器代码前添加函数实现
      addFunction(name, func.implementation);
    }
  }
  
  // 如果需要添加uTime但没有
  if (!modifiedCode.includes("uniform float uTime") && 
      (modifiedCode.includes("uTime") || modifiedCode.toLowerCase().includes("time"))) {
    modifiedCode = `uniform float uTime;\n${modifiedCode}`;
    madeChanges = true;
  }
  
  // 检查是否缺少varying vUv
  if (modifiedCode.includes("vUv") && !modifiedCode.includes("varying vec2 vUv")) {
    modifiedCode = `varying vec2 vUv;\n${modifiedCode}`;
    madeChanges = true;
  }
  
  return madeChanges ? modifiedCode : shaderCode;
};

export default function ShaderGenerator() {
  const [vertexShader, setVertexShader] = useState(DEFAULT_VERTEX_SHADER)
  const [fragmentShader, setFragmentShader] = useState(DEFAULT_FRAGMENT_SHADER)
  const [activeTab, setActiveTab] = useState("fragment")
  const [geometryType, setGeometryType] = useState<"sphere" | "box" | "plane">("sphere")
  const [inputMessage, setInputMessage] = useState("")
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const orbitControlsRef = useRef(null)
  const [historySummary, setHistorySummary] = useState("Initial session");
  const [shaderError, setShaderError] = useState<string | null>(null);
  const agentTools = [new TavilySearchResults({ 
    maxResults: 3,
    apiKey: process.env.NEXT_PUBLIC_TAVILY_API_KEY // 添加API key配置
  })];
  const [selectedModel, setSelectedModel] = useState<"gpt-4o" | "sonnet" | "deepseek">("gpt-4o")
  const [chatMessages, setChatMessages] = useState<
  { role: "user" | "assistant"; content: string }[]
>([
  { 
    role: "assistant", 
    content: "Welcome to Pelote - Online Shader Generator! I can help you create and optimize shaders. What can I help you with?" 
  },
]);
  
const agentCheckpointer = useRef(new SimpleMemorySaver());
const agent = useRef(createReactAgent({
  llm: new ChatOpenAI({ 
    temperature: 0,
    openAIApiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY,
    modelName: "gpt-4o"
  }),
  tools: agentTools,

}));

  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [chatMessages])

  const handleCodeChange = (code: string) => {
    if (activeTab === "vertex") {
      setVertexShader(code)
    } else {
      setFragmentShader(code)
    }
  }

  const compileAndRefresh = () => {
    // 新增空内容检查
    const currentShader = activeTab === "vertex" ? vertexShader : fragmentShader;
    if (!currentShader.trim()) {
      alert("Shader代码不能为空！");
      return;
    }
    
    try {
      // 检查是否包含uTime变量
      if (activeTab === "fragment" && !fragmentShader.includes("uniform float uTime")) {
        // 如果没有uTime声明，自动添加
        setFragmentShader((prev) => {
          const lines = prev.split("\n");
          // 在第一个非空行之后添加uTime声明
          for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim() !== "") {
              lines.splice(i + 1, 0, "uniform float uTime;");
              break;
            }
          }
          return lines.join("\n");
        });
      }
      
      // 更可靠的编译触发方法
      if (activeTab === "vertex") {
        // 为顶点着色器添加注释标记以触发重新编译
        setVertexShader((prev) => `${prev.trim()}\n// Recompile trigger: ${Date.now()}`);
      } else {
        // 为片段着色器添加注释标记以触发重新编译
        setFragmentShader((prev) => `${prev.trim()}\n// Recompile trigger: ${Date.now()}`);
      }
      
      // 设置成功消息
      console.log("着色器编译已触发");
    } catch (error) {
      // 捕获任何可能的错误
      console.error("编译过程中出错:", error);
      setShaderError(`编译错误: ${error instanceof Error ? error.message : String(error)}`);
      
      // 尝试恢复到默认状态
      if (error instanceof Error && error.message.includes("严重错误")) {
        if (confirm("着色器出现严重错误，是否重置为默认着色器？")) {
          createNewShader();
        }
      }
    }
  };

  const createNewShader = () => {
    setVertexShader(DEFAULT_VERTEX_SHADER)
    setFragmentShader(DEFAULT_FRAGMENT_SHADER)
  }

  const downloadShader = () => {
    const element = document.createElement("a")
    const file = new Blob([fragmentShader], { type: "text/plain" })
    element.href = URL.createObjectURL(file)
    element.download = "shader.glsl"
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }
  

  const current_shader = activeTab === "vertex" ? vertexShader : fragmentShader;
  const user_message = inputMessage; // 直接使用已有状态

  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const newMessages = [
      ...chatMessages, 
      { role: "user" as const, content: inputMessage }
    ];
    setChatMessages(newMessages);
    setChatMessages([...newMessages, { role: "assistant" as const, content: "..." }]);
    setIsLoading(true);
    const userMessageCopy = inputMessage;
    setInputMessage("");

    try {
      // 动态创建 agentModel 和 agent
      let agentModelInstance: any = null;
      if (selectedModel === "gpt-4o") {
        agentModelInstance = new ChatOpenAI({ 
          temperature: 0,
          openAIApiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY,
          modelName: "gpt-4o"
        });
      } else if (selectedModel === "sonnet") {
        agentModelInstance = new ChatOpenAI({ 
          temperature: 0,
          openAIApiKey: process.env.NEXT_PUBLIC_SONNET_API_KEY,
          modelName: "sonnet"
        });
      } else if (selectedModel === "deepseek") {
        agentModelInstance = new ChatOpenAI({ 
          temperature: 0,
          openAIApiKey: process.env.NEXT_PUBLIC_DEEPSEEK_API_KEY,
          modelName: "deepseek"
        });
      }
      if (!agentModelInstance) {
        setChatMessages(prev => {
          return [...prev.slice(0, prev.length - 1), { role: "assistant", content: "模型选择无效，请重试。" }];
        });
        setIsLoading(false);
        return;
      }
      const agentInstance = createReactAgent({
        llm: agentModelInstance,
        tools: agentTools,
      });

      const response = await agentInstance.invoke({
          messages: [
            {
              role: "system",
              content: `
              你是顶尖GLSL视觉艺术家，遵循以下核心原则：
              - 动态美学：效果随时间演变，至少包含2个相位变化参数
              - 非线性插值：使用smoothstep/cubic，禁止线性变化
              - 多维复合：组合至少3种基础技术
              - 色彩规范：使用HSL/HSV空间，禁止直接操作RGB

              噪声库技术：
              1. Perlin噪声 - float pnoise(vec2/vec3 p, float freq)
                  用途：云彩、火焰、平滑有机纹理；特性：渐变平滑，易分形叠加

              2. Simplex噪声 - float snoise(vec2/vec3 p)
                  用途：复杂流体、电子场；特性：高效、维度扩展性好

              3. Cellular噪声 - vec2 cellular(vec2 p)
                  用途：细胞、晶格结构；特性：蜂窝状分布，自然边界

              4. FBM技术 - float fbm(vec2 p) { return noise(p) + 0.5*noise(p*2.1) + 0.25*noise(p*4.3); }
                  用途：山脉、云层；特性：多尺度细节，自相似性

              5. 域翘曲 - vec2 q = p + vec2(noise(p+t), noise(p-t))
                  用途：熔岩灯、流体；特性：空间扭曲，增强流动感

              组合技术：
              - 透明度混合：mix(noise1, noise2, smoothstep(0.4, 0.6, noise3))
              - 递归噪声：noise(p + noise(p*2.0))
              - 梯度域：vec2(noise(p+dx)-noise(p-dx), noise(p+dy)-noise(p-dy))

              创意方向：
              - 复数域变换：z^2+c 有机图案
              - 场效应叠加：辐射场/涡旋场/晶格场
              - 光效层：bloom光晕（模糊+衍射）
              - 流体模拟：简化Navier-Stokes
              - 生物形态：L-system图案生长
              - 粒子系统：Euler积分

              输入结构：
              - current_shader: 当前代码文本
              - user_message: 对话内容
              - history_summary: 历史摘要

              输出格式：
              {
                "new_shader_code": "完整的GLSL代码",
                "user_response": "分步骤解释（支持Markdown）",
                "new_history_summary": "精简版修改记录"
              }

              要求：
              - 返回纯JSON格式，不要包含json等代码块标记
              - 不要在JSON前后添加任何文本
              - 代码保持语法完整
              `
            },
            {
              role: "user",
              content: `
              - current_shader: ${current_shader}
              - user_message: ${userMessageCopy}
              - history_summary: ${historySummary}`
            }
          ]
        }
      , { configurable: { thread_id: "shader_thread" } });
      
      const aiMessage = response.messages.find(m => m._getType() === "ai");
      if (aiMessage && typeof aiMessage.content === "string") {
        try {
          // 清理 Markdown 格式并提取 JSON
          let contentToParse = aiMessage.content;
          
          // 记录原始内容用于调试
          console.log("原始响应内容:", contentToParse);
          
          // 尝试多种解析策略
          let responseData = null;
          
          // 策略1: 尝试直接解析整个内容
          try {
            responseData = JSON.parse(contentToParse);
          } catch (e) {
            console.log("直接解析失败，尝试清理格式...");
            
            // 策略2: 移除Markdown代码块标记
            const cleanContent = contentToParse
              .replace(/```(json|javascript|glsl)?[\r\n]*/g, '') // 移除开始标记
              .replace(/```[\r\n]*/g, '');                       // 移除结束标记
            
            try {
              responseData = JSON.parse(cleanContent);
            } catch (e) {
              console.log("清理后解析失败，尝试提取JSON对象...");
              
              // 策略3: 使用正则提取最外层的JSON对象
              const jsonRegex = /{[\s\S]*?}/g;
              const jsonMatches = cleanContent.match(jsonRegex);
              
              if (jsonMatches && jsonMatches.length > 0) {
                // 尝试解析找到的第一个完整JSON对象
                try {
                  responseData = JSON.parse(jsonMatches[0]);
                } catch (e) {
                  // 尝试解析其他匹配项
                  for (let i = 1; i < jsonMatches.length; i++) {
                    try {
                      const parsed = JSON.parse(jsonMatches[i]);
                      if (parsed && 
                          (parsed.new_shader_code || 
                           parsed.user_response || 
                           parsed.new_history_summary)) {
                        responseData = parsed;
                        break; // 找到有效的响应就退出
                      }
                    } catch (innerE) {
                      continue; // 继续尝试下一个
                    }
                  }
                }
              }
              
              // 如果还是没有匹配，尝试更宽松的匹配
              if (!responseData) {
                console.log("标准JSON提取失败，尝试修复JSON格式...");
                
                // 策略4: 尝试手动提取关键部分
                const shaderCodeMatch = cleanContent.match(/"new_shader_code":\s*"([\s\S]*?)(?<!\\)"/);
                const userResponseMatch = cleanContent.match(/"user_response":\s*"([\s\S]*?)(?<!\\)"/);
                const historyMatch = cleanContent.match(/"new_history_summary":\s*"([\s\S]*?)(?<!\\)"/);
                
                if (shaderCodeMatch || userResponseMatch) {
                  responseData = {
                    new_shader_code: shaderCodeMatch ? shaderCodeMatch[1].replace(/\\n/g, '\n').replace(/\\"/g, '"') : null,
                    user_response: userResponseMatch ? userResponseMatch[1].replace(/\\n/g, '\n').replace(/\\"/g, '"') : "解析到着色器代码但未解析到响应文本",
                    new_history_summary: historyMatch ? historyMatch[1].replace(/\\n/g, '\n').replace(/\\"/g, '"') : historySummary
                  };
                }
              }
            }
          }
          
          // 如果所有策略都失败了
          if (!responseData) {
            throw new Error("无法从响应中提取有效的JSON数据");
          }
          
          console.log("成功解析的数据类型:", typeof responseData);
          
          // 检查shader代码是否包含未实现的函数声明
          if (responseData.new_shader_code) {
            // 应用更强大的依赖检测和修复
            responseData.new_shader_code = checkShaderDependencies(responseData.new_shader_code);
            
            // 确保着色器代码末尾没有多余的空行和注释，这可能导致编译问题
            responseData.new_shader_code = responseData.new_shader_code.trim();
            
            // 处理响应...
            setHistorySummary(responseData.new_history_summary || historySummary);
            setChatMessages(prev => prev.slice(0, prev.length - 1));
            setChatMessages(prev => [...prev, { role: "assistant" as const, content: "" }]);
            
            // 使用伪流式输出
            if (responseData.user_response) {
              simulateTypingEffect(
                responseData.user_response,
                (text) => {
                  setChatMessages(prev => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1].content = text;
                    return newMessages;
                  });
                },
                30
              );
            }
            
            // 更新shader代码
            if (responseData.new_shader_code) {
              if (activeTab === "vertex") {
                setVertexShader(responseData.new_shader_code);
              } else {
                setFragmentShader(responseData.new_shader_code);
                // 触发刷新
                setTimeout(() => compileAndRefresh(), 100);
              }
            }
          }
        } catch (e) {
          console.error('JSON解析失败:', e);
          setChatMessages(prev => {
            return [...prev.slice(0, prev.length - 1), { 
              role: "assistant", 
              content: "响应解析失败。请尝试简化您的请求或刷新页面。" 
            }];
          });
        }
      }
    } catch (error) {
      console.error("调用错误:", error);
      
      // 如果API调用失败，替换加载指示器为错误消息
      setChatMessages(prev => {
        return [...prev.slice(0, prev.length - 1), { role: "assistant", content: "连接错误，请检查API密钥和网络" }];
      });
    } finally {
      setIsLoading(false);
    }
  };

  // 添加模拟打字效果的函数
  const simulateTypingEffect = async (
    fullText: string, 
    setDisplayFunction: (text: string) => void, 
    typingSpeed = 20
  ): Promise<void> => {
    let currentText = "";
    const words = fullText.split(" ");
    
    for (let i = 0; i < words.length; i++) {
      currentText += words[i] + (i < words.length - 1 ? " " : "");
      setDisplayFunction(currentText);
      await new Promise(resolve => setTimeout(resolve, typingSpeed));
    }
  };

  // 添加一个加载指示器组件
  const LoadingDots = () => {
    const [dots, setDots] = useState(".");
    
    useEffect(() => {
      const interval = setInterval(() => {
        setDots(prev => {
          if (prev === "...") return ".";
          return prev + ".";
        });
      }, 500);
      
      return () => clearInterval(interval);
    }, []);
    
    return <span>{dots}</span>;
  };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-black to-slate-900">
      {/* Header */}
      <header className="p-4 flex items-center justify-between border-b border-slate-800">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center gap-3"
      >
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center">
          <img 
            src="/latent-cat-logo.svg" 
            className="w-8 h-8 rounded-full object-cover"
            alt="Latent Cat Logo"
          />
        </div>
        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-cyan-400">
        Pelote - Online Shader Generator
        </h1>
      </motion.div>

        <div className="flex gap-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-4 py-2 rounded-full bg-slate-800 text-white text-sm hover:bg-slate-700 transition-colors"
          >
            Log in
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-4 py-2 rounded-full bg-gradient-to-r from-purple-500 to-cyan-500 text-white text-sm hover:from-purple-600 hover:to-cyan-600 transition-colors"
          >
            Share
          </motion.button>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex flex-col md:flex-row gap-4 p-4 h-100vh">
        {/* Left Panel - Shader Preview */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="w-full md:w-1/3 flex flex-col gap-4"
        >
          <div className="relative aspect-square rounded-2xl overflow-hidden bg-black border border-slate-800 shadow-xl shadow-purple-900/20">
            <ShaderPreview 
              vertexShader={vertexShader} 
              fragmentShader={fragmentShader} 
              geometryType={geometryType}
              onError={(errorMsg: string) => {
                console.error("Shader编译错误:", errorMsg);
                setShaderError(errorMsg);
                
                // 如果是片段着色器错误，显示在UI中
                if (errorMsg.includes("Fragment shader") && !errorMsg.includes("Vertex shader")) {
                  setChatMessages(prev => {
                    // 查找最后一个助手消息并追加错误提示
                    const newMessages = [...prev];
                    for (let i = newMessages.length - 1; i >= 0; i--) {
                      if (newMessages[i].role === "assistant") {
                        newMessages[i].content += `\n\n⚠️ **着色器编译错误**：\n\`\`\`\n${errorMsg}\n\`\`\``;
                        break;
                      }
                    }
                    return newMessages;
                  });
                }
              }} 
            />

            {/* Geometry Type Selector */}
            <div className="absolute top-4 right-4 flex gap-2">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => {
                  setGeometryType("sphere");
                  setTimeout(() => compileAndRefresh(), 50);
                }}
                className={`w-10 h-10 rounded-full flex items-center justify-center backdrop-blur-sm ${
                  geometryType === "sphere" ? "bg-purple-500 text-white" : "bg-black/50 text-white/70 hover:bg-black/70"
                }`}
              >
                <Sphere size={18} />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => {
                  setGeometryType("box");
                  setTimeout(() => compileAndRefresh(), 50);
                }}
                className={`w-10 h-10 rounded-full flex items-center justify-center backdrop-blur-sm ${
                  geometryType === "box" ? "bg-purple-500 text-white" : "bg-black/50 text-white/70 hover:bg-black/70"
                }`}
              >
                <Cube size={18} />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => {
                  setGeometryType("plane");
                  setTimeout(() => compileAndRefresh(), 50);
                }}
                className={`w-10 h-10 rounded-full flex items-center justify-center backdrop-blur-sm ${
                  geometryType === "plane" ? "bg-purple-500 text-white" : "bg-black/50 text-white/70 hover:bg-black/70"
                }`}
              >
                <Square size={18} />
              </motion.button>
            </div>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={createNewShader}
              className="flex-none flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-slate-800 text-white hover:bg-slate-700 transition-all min-w-[120px]"
            >
              <Plus size={16} className="shrink-0" />
              <span className="text-sm whitespace-nowrap">New</span>
            </motion.button>


            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={compileAndRefresh}
              className="flex-none flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-slate-800 text-white hover:bg-slate-700 transition-all min-w-[150px]"
            >
              <RefreshCw size={16} className="shrink-0" />
              <span className="text-sm whitespace-nowrap">Refresh Shader</span>
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={downloadShader}
              className="flex-none flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gradient-to-r from-purple-500 to-cyan-500 text-white hover:from-purple-600 hover:to-cyan-600 transition-all  min-w-[120px]"
            >
              <Download size={16} className="shrink-0" />
              <span className="text-sm whitespace-nowrap">Download</span>
            </motion.button>
          </div>
        </motion.div>

        {/* Middle Panel - Code Editor */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="w-full md:w-1/3 flex flex-col h-[calc(100vh-8rem)]"
        >
          <div className="flex mb-2">
            <button
              onClick={() => setActiveTab("fragment")}
              className={`px-4 py-2 rounded-t-lg ${
                activeTab === "fragment"
                  ? "bg-slate-800 text-white"
                  : "bg-slate-900 text-slate-400 hover:bg-slate-800/50"
              }`}
            >
              Fragment Shader
            </button>
            <button
              onClick={() => setActiveTab("vertex")}
              className={`px-4 py-2 rounded-t-lg ${
                activeTab === "vertex" ? "bg-slate-800 text-white" : "bg-slate-900 text-slate-400 hover:bg-slate-800/50"
              }`}
            >
              Vertex Shader
            </button>
          </div>

          <div className="flex-1 rounded-lg overflow-hidden border border-slate-800 shadow-xl shadow-purple-900/20">
            <ShaderCodeEditor
              value={activeTab === "vertex" ? vertexShader : fragmentShader}
              onChange={handleCodeChange}
              language="glsl"
            />
          </div>
        </motion.div>

        {/* Right Panel - Chat */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="w-full md:w-1/3 flex flex-col rounded-lg border border-slate-800 shadow-xl shadow-purple-900/20 bg-slate-900/50 backdrop-blur-sm h-[calc(100vh-8rem)]"
        >
          <div className="p-4 border-b border-slate-800">
            <h2 className="text-xl font-bold text-white">Dialog</h2>
            <div className="flex items-center gap-2 mt-2 mb-2">
              <span className="text-slate-400 text-sm">选择模型:</span>
              <select
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value as any)}
                className="bg-slate-800 text-white rounded px-2 py-1 text-sm border border-slate-700 focus:outline-none"
              >
                <option value="gpt-4o">GPT-4o</option>
                <option value="sonnet">Sonnet</option>
                <option value="deepseek">DeepSeek</option>
              </select>
            </div>
          </div>

          <div
            ref={chatContainerRef}
            className="flex-1 p-4 overflow-y-auto"
          >
            <AnimatePresence>
              {chatMessages.map((message, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`mb-4 flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-2 ${
                      message.role === "user"
                        ? "bg-gradient-to-r from-purple-500 to-cyan-500 text-white"
                        : "bg-slate-800 text-white"
                    }`}
                  >
                    {message.content === "..." && isLoading ? (
                      <LoadingDots />
                    ) : (
                      <div className="markdown-content">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({ node, ...props }: any) => <p className="mb-2" {...props} />,
                            a: ({ node, ...props }: any) => <a className="text-cyan-400 underline" {...props} />,
                            h1: ({ node, ...props }: any) => <h1 className="text-xl font-bold mb-2 mt-4" {...props} />,
                            h2: ({ node, ...props }: any) => <h2 className="text-lg font-bold mb-2 mt-3" {...props} />,
                            h3: ({ node, ...props }: any) => <h3 className="text-md font-bold mb-1 mt-2" {...props} />,
                            ul: ({ node, ...props }: any) => <ul className="list-disc pl-5 mb-2" {...props} />,
                            ol: ({ node, ...props }: any) => <ol className="list-decimal pl-5 mb-2" {...props} />,
                            li: ({ node, ...props }: any) => <li className="mb-1" {...props} />,
                            pre: ({ node, ...props }: any) => (
                              <pre className="bg-slate-700 p-2 rounded my-2 overflow-auto" {...props} />
                            ),
                            code: ({ node, inline, className, children, ...props }: any) => {
                              if (inline) {
                                return (
                                  <code className="bg-slate-700 px-1 rounded text-cyan-300" {...props}>
                                    {children}
                                  </code>
                                );
                              }
                              return (
                                <code className="text-cyan-300" {...props}>
                                  {children}
                                </code>
                              );
                            },
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          <div className="p-4 border-t border-slate-800">
            <div className="flex gap-2">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Type your request..."
                className="flex-1 bg-slate-800 border border-slate-700 rounded-full px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={sendMessage}
                className="w-10 h-10 rounded-full bg-gradient-to-r from-purple-500 to-cyan-500 flex items-center justify-center text-white"
              >
                <Send size={18} />
              </motion.button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
// 修改 ShaderCodeEditor 组件
function ShaderCodeEditor({
  value,
  onChange,
  language,
}: { value: string; onChange: (value: string) => void; language: string }) {
  return (
    <div className="w-full h-full overflow-hidden">
      <CodeEditorComponent
        value={value}
        language={language}
        onChange={(evn) => {
          // 确保从事件对象中获取值并传递给父组件的onChange函数
          if (evn && evn.target) {
            onChange(evn.target.value);
          }
        }}
        padding={16}
        style={{
          fontSize: 14,
          backgroundColor: "#1e1e1e",
          fontFamily: 
            'ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace',
          height: "100%", // 使用100%高度
          overflow: "auto", // 添加滚动条
        }}
        data-color-mode="dark"
        placeholder="// 输入你的 GLSL 代码..."
      />
    </div>
  );
}

