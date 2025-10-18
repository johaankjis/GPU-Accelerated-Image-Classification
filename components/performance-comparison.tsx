"use client"

import { Card } from "@/components/ui/card"
import { Bar, BarChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export function PerformanceComparison() {
  const performanceData = [
    {
      metric: "Training Time",
      cpu: 245,
      gpu: 159,
      unit: "seconds",
    },
    {
      metric: "Inference Time",
      cpu: 18.5,
      gpu: 13.2,
      unit: "ms/batch",
    },
    {
      metric: "Throughput",
      cpu: 54,
      gpu: 76,
      unit: "samples/sec",
    },
  ]

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Performance Comparison</h2>
          <p className="text-sm text-muted-foreground mt-1">CPU vs GPU acceleration metrics</p>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-slate-500" />
            <span className="text-muted-foreground">CPU</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-emerald-500" />
            <span className="text-muted-foreground">GPU</span>
          </div>
        </div>
      </div>

      <ChartContainer
        config={{
          cpu: {
            label: "CPU",
            color: "hsl(var(--chart-3))",
          },
          gpu: {
            label: "GPU",
            color: "hsl(var(--chart-2))",
          },
        }}
        className="h-[300px]"
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
            <XAxis dataKey="metric" stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} tickLine={false} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="cpu" fill="hsl(var(--chart-3))" radius={[4, 4, 0, 0]} />
            <Bar dataKey="gpu" fill="hsl(var(--chart-2))" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>

      <div className="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-border">
        <div>
          <p className="text-xs text-muted-foreground">Training Speedup</p>
          <p className="text-xl font-semibold text-emerald-500 mt-1">35.2% faster</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Inference Speedup</p>
          <p className="text-xl font-semibold text-emerald-500 mt-1">28.7% faster</p>
        </div>
      </div>
    </Card>
  )
}
