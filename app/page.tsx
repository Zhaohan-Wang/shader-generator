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
  const agentModel = new ChatOpenAI({ 
    temperature: 0,
    openAIApiKey: process.env.NEXT_PUBLIC_OPENAI_API_KEY,
    modelName: "gpt-4o"
  });
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
  llm: agentModel,
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

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const newMessages = [
      ...chatMessages, 
      { 
        role: "user" as const,  // 使用as const明确类型
        content: inputMessage 
      }
    ];
    setChatMessages(newMessages);
    setInputMessage("");


    try {
      const response = await agent.current.invoke(
        { 
          messages: [
            {
              role: "system",
              content: `
              你是一名顶尖的GLSL视觉艺术家，遵循以下创作原则：
              核心设计准则
                动态美学优先：默认使用片段着色器，所有效果必须随时间演变，至少包含2个相位变化参数
                有机运动法则：强制使用非线性插值（smoothstep/cubic），禁止线性变化
                多维复合：必须组合至少3种基础技术（如分形+极坐标+光线追踪）
                色彩美学：默认使用HSL色彩空间，禁止直接操作RGB三元色
                能量守恒：颜色空间必须使用HSL/HSV，禁止直接操作RGB三元色
              可选技术工具与创意思路：
                噪声矩阵：Perlin/SIMPLEX噪声作为基底纹理
                复数域变换：运用复平面旋转(z^2+c)生成有机图案
                场效应叠加：至少包含辐射场/涡旋场/晶格场中的两种
                光效层：添加bloom光晕（使用指数模糊+色彩衍射）
                超现实流体：结合Navier-Stokes方程简化模型
                量子化效应：引入概率波函数坍缩可视化
                生物形态生成：使用L-system语法驱动图案生长
                时空扭曲：整合洛伦兹变换因子
                粒子系统：结合Euler积分模拟
              输入结构：
              - current_shader: 当前 shader 代码文本
              - user_message: 本次对话内容
              - history_summary: 历史摘要

              输出结构：
              直接返回包含了所有数据的 json，前后不需要说话，返回的 json 需要直接能够被序列化，前后不能添加 markdown 标记。
              {
                "new_shader_code": "string // 完整的新GLSL代码（包含所有修改）",
                "user_response": "string // 给用户的文字回复（分步骤说明修改/解释）",
                "new_history_summary": "string // 更新后的历史摘要（精简版修改记录）"
              }
              响应要求：
                新代码必须保持语法完整可直接运行
                自动添加时间变量uTime
              `
            },
            {
              role: "user",
              content: `
              - current_shader: ${current_shader}
              - user_message: ${user_message}
              - history_summary: ${historySummary}`
            }
          ] 
        },
        { configurable: { thread_id: "shader_thread" } }
      );
      console.log('原始响应对象:', response);
      console.log('响应消息列表:', response.messages);
      
      const aiMessage = response.messages.find(m => m._getType() === "ai");
      if (aiMessage && typeof aiMessage.content === "string") {
        try {
          const responseData = JSON.parse(
            // 修复多个JSON对象问题，取第一个有效对象
            aiMessage.content.split(/(?<=})\s*(?={)/)[0] // 使用正则分割多个JSON对象
          );
          console.log('解析后的JSON:', responseData);
          
          // 更新历史摘要状态
          setHistorySummary(responseData.new_history_summary || historySummary);
          
          // 只显示user_response内容
          setChatMessages(prev => [
            ...prev,
            { 
              role: "assistant" as const,
              content: responseData.user_response || "收到空响应"
            }
          ]);
          
          // 更新shader代码逻辑保持不变
          if (responseData.new_shader_code) {
            if (activeTab === "vertex") {
              setVertexShader(responseData.new_shader_code);
            } else {
              setFragmentShader(responseData.new_shader_code);
            }
          }
        } catch (e) {
          console.error('JSON解析失败，原始内容:', aiMessage.content);
          console.error('错误详情:', e);
          // 添加错误提示
          setChatMessages(prev => [
            ...prev,
            { 
              role: "assistant" as const,
              content: "响应解析失败，请检查AI返回格式"
            }
          ]);
        }
      }

    } catch (error) {
      console.error("Agent 调用错误:", error);
    }
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
            <ShaderPreview vertexShader={vertexShader} fragmentShader={fragmentShader} geometryType={geometryType} />

            {/* Geometry Type Selector */}
            <div className="absolute top-4 right-4 flex gap-2">
              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setGeometryType("sphere")}
                className={`w-10 h-10 rounded-full flex items-center justify-center backdrop-blur-sm ${
                  geometryType === "sphere" ? "bg-purple-500 text-white" : "bg-black/50 text-white/70 hover:bg-black/70"
                }`}
              >
                <Sphere size={18} />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setGeometryType("box")}
                className={`w-10 h-10 rounded-full flex items-center justify-center backdrop-blur-sm ${
                  geometryType === "box" ? "bg-purple-500 text-white" : "bg-black/50 text-white/70 hover:bg-black/70"
                }`}
              >
                <Cube size={18} />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => setGeometryType("plane")}
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
          className="w-full md:w-1/3 flex flex-col"
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
          className="w-full md:w-1/3 flex flex-col rounded-lg border border-slate-800 shadow-xl shadow-purple-900/20 bg-slate-900/50 backdrop-blur-sm"
        >
          <div className="p-4 border-b border-slate-800">
            <h2 className="text-xl font-bold text-white">Dialog</h2>
          </div>

          <div
            ref={chatContainerRef}
            className="flex-1 p-4 overflow-y-auto"
            style={{ maxHeight: "calc(100vh - 13rem)" }}
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
                    {message.content}
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