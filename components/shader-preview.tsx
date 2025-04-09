"use client"

import { useRef, useEffect } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { OrbitControls, PerspectiveCamera, Grid } from "@react-three/drei"
import * as THREE from "three"
import { useSpring, animated } from "@react-spring/three"

interface ShaderPreviewProps {
  vertexShader: string
  fragmentShader: string
  geometryType: "sphere" | "box" | "plane"
  onError?: (errorMsg: string) => void
}

export default function ShaderPreview({ vertexShader, fragmentShader, geometryType, onError }: ShaderPreviewProps) {
  return (
    <Canvas>
      <PerspectiveCamera makeDefault position={[0, 0, 2.5]} />
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={1.5}
        maxDistance={4}
        autoRotate
        autoRotateSpeed={0.5}
      />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <ShaderMesh vertexShader={vertexShader} fragmentShader={fragmentShader} geometryType={geometryType} onError={onError} />
      <Environment />
    </Canvas>
  )
}

function ShaderMesh({
  vertexShader,
  fragmentShader,
  geometryType,
  onError,
}: {
  vertexShader: string
  fragmentShader: string
  geometryType: "sphere" | "box" | "plane"
  onError?: (errorMsg: string) => void
}) {
  const meshRef = useRef<THREE.Mesh>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)
  const clock = new THREE.Clock()

  // Create animated rotation
  const [spring, api] = useSpring(() => ({
    rotation: [0, 0, 0],
    config: { mass: 2, tension: 200, friction: 30 },
  }))

  // Handle mouse hover effect
  const handlePointerOver = () => {
    api.start({ rotation: [0, Math.PI / 4, 0] })
  }

  const handlePointerOut = () => {
    api.start({ rotation: [0, 0, 0] })
  }

  // Update shader uniforms
  useFrame(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = clock.getElapsedTime()
    }
  })

  // Create a reusable shader material
  useEffect(() => {
    if (!materialRef.current) return;
    
    try {
      // Attempt to compile the shader
      const tempMaterial = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: materialRef.current.uniforms
      });
      
      // If we get here without error, update the actual material
      materialRef.current.vertexShader = vertexShader;
      materialRef.current.fragmentShader = fragmentShader;
      materialRef.current.needsUpdate = true;
    } catch (error) {
      console.error("Shader compilation error:", error);
      if (onError) {
        onError(error instanceof Error ? error.message : String(error));
      }
    }
  }, [vertexShader, fragmentShader, onError]);

  // Render the appropriate geometry based on the selected type
  const renderGeometry = () => {
    switch (geometryType) {
      case "sphere":
        return <sphereGeometry args={[1, 64, 64]} />
      case "box":
        return <boxGeometry args={[1.5, 1.5, 1.5]} />
      case "plane":
        return <planeGeometry args={[2, 2, 32, 32]} />
      default:
        return <sphereGeometry args={[1, 64, 64]} />
    }
  }

  return (
    <animated.mesh
      ref={meshRef}
      rotation={spring.rotation as any}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
    >
      {renderGeometry()}
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        side={THREE.DoubleSide}
        uniforms={{
          uTime: { value: 0 },
          uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
          uMouse: { value: new THREE.Vector2(0, 0) },
        }}
      />
    </animated.mesh>
  )
}

function Environment() {
  return (
    <>
      {/* 3D Grid to emphasize the 3D space */}
      <Grid
        position={[0, -1.5, 0]}
        args={[10, 10]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#6366f1"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#8b5cf6"
        fadeDistance={30}
        fadeStrength={1}
      />

      {/* Subtle fog for depth */}
      <fog attach="fog" args={["#000", 5, 15]} />

      {/* Axis helpers for better orientation */}
      <AxisHelpers />
    </>
  )
}

function AxisHelpers() {
  const { scene } = useThree()

  useEffect(() => {
    const axisHelper = new THREE.AxesHelper(5)
    axisHelper.position.set(-5, -1.5, -5)
    scene.add(axisHelper)

    return () => {
      scene.remove(axisHelper)
    }
  }, [scene])

  return null
}

