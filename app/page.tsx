import { TrainingMetrics } from "@/components/training-metrics"
import { InferenceResults } from "@/components/inference-results"
import { PerformanceComparison } from "@/components/performance-comparison"
import { ModelOverview } from "@/components/model-overview"

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-semibold text-foreground">GPU Image Classification</h1>
              <p className="text-sm text-muted-foreground mt-1">CIFAR-10 CNN Training Dashboard</p>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/20">
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <span className="text-sm font-medium text-emerald-500">GPU Active</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="grid gap-6">
          <ModelOverview />

          <div className="grid lg:grid-cols-2 gap-6">
            <TrainingMetrics />
            <PerformanceComparison />
          </div>

          <InferenceResults />
        </div>
      </main>
    </div>
  )
}
