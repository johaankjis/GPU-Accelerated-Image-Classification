"use client"

import { Card } from "@/components/ui/card"
import { Line, LineChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export function TrainingMetrics() {
  // Mock training data
  const trainingData = Array.from({ length: 50 }, (_, i) => ({
    epoch: i + 1,
    accuracy: Math.min(92.4, 45 + i * 1.2 + Math.random() * 3),
    loss: Math.max(0.15, 2.5 - i * 0.05 + Math.random() * 0.1),
    gpuUtil: 85 + Math.random() * 10,
  }))

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Training Progress</h2>
          <p className="text-sm text-muted-foreground mt-1">Accuracy and loss over 50 epochs</p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-muted-foreground">Accuracy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500" />
            <span className="text-muted-foreground">Loss</span>
          </div>
        </div>
      </div>

      <ChartContainer
        config={{
          accuracy: {
            label: "Accuracy",
            color: "hsl(var(--chart-1))",
          },
          loss: {
            label: "Loss",
            color: "hsl(var(--chart-4))",
          },
        }}
        className="h-[300px]"
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Line type="monotone" dataKey="accuracy" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="loss" stroke="hsl(var(--chart-4))" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>

      <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-border">
        <div>
          <p className="text-xs text-muted-foreground">Final Accuracy</p>
          <p className="text-xl font-semibold text-foreground mt-1">92.4%</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Final Loss</p>
          <p className="text-xl font-semibold text-foreground mt-1">0.187</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Avg GPU Util</p>
          <p className="text-xl font-semibold text-foreground mt-1">89.3%</p>
        </div>
      </div>
    </Card>
  )
}
