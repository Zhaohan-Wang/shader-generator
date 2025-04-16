# Pelote - Online Shader Generator

Pelote 是一个在线 GLSL 着色器生成器，支持实时预览、模型对话（支持 GPT-4o、Sonnet、DeepSeek）以及代码编辑。

## 功能特点

- 实时预览 GLSL 着色器效果
- 支持顶点着色器和片段着色器编辑
- 提供多种几何体预览（球体、立方体、平面）
- 内置噪声函数库（Perlin、Simplex、Cellular、FBM 等）
- 支持与不同大模型对话（GPT-4o、Sonnet、DeepSeek）
- 支持下载生成的着色器代码

## 安装与部署

### 环境要求

- Node.js >= 18
- pnpm >= 8

### 安装步骤

1. 克隆项目

```bash
git clone https://github.com/yourusername/shader-generator.git
cd shader-generator
```

2. 安装依赖

```bash
pnpm install
```

3. 配置环境变量

在项目根目录创建 `.env.local` 文件，并填入以下内容：

```env
NEXT_PUBLIC_OPENAI_API_KEY=your_openai_api_key
NEXT_PUBLIC_SONNET_API_KEY=your_sonnet_api_key
NEXT_PUBLIC_DEEPSEEK_API_KEY=your_deepseek_api_key
NEXT_PUBLIC_TAVILY_API_KEY=your_tavily_api_key
```

4. 启动开发服务器

```bash
pnpm dev
```

5. 构建生产版本

```bash
pnpm build
pnpm start
```

## 使用说明

1. 在左侧面板可以预览着色器效果，并切换不同的几何体
2. 中间面板可以编辑顶点着色器和片段着色器代码
3. 右侧面板可以与大模型对话，生成或优化着色器代码
4. 点击 "Refresh Shader" 按钮可以重新编译着色器
5. 点击 "Download" 按钮可以下载当前着色器代码

## 技术栈

- Next.js
- React
- Three.js
- LangChain
- Tailwind CSS
- Framer Motion

## 许可证

MIT 