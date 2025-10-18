import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import Image from "next/image"

export function InferenceResults() {
  const predictions = [
    {
      id: 1,
      image: "/airplane-in-flight.png",
      predicted: "Airplane",
      confidence: 98.2,
      actual: "Airplane",
      correct: true,
    },
    {
      id: 2,
      image: "/automobile.jpg",
      predicted: "Automobile",
      confidence: 95.7,
      actual: "Automobile",
      correct: true,
    },
    {
      id: 3,
      image: "/colorful-bird-perched.png",
      predicted: "Bird",
      confidence: 89.3,
      actual: "Bird",
      correct: true,
    },
    {
      id: 4,
      image: "/tabby-cat-sunbeam.png",
      predicted: "Cat",
      confidence: 92.1,
      actual: "Cat",
      correct: true,
    },
    {
      id: 5,
      image: "/majestic-deer.png",
      predicted: "Deer",
      confidence: 87.5,
      actual: "Deer",
      correct: true,
    },
    {
      id: 6,
      image: "/happy-golden-retriever.png",
      predicted: "Dog",
      confidence: 94.8,
      actual: "Dog",
      correct: true,
    },
    {
      id: 7,
      image: "/green-frog-on-lilypad.png",
      predicted: "Frog",
      confidence: 91.2,
      actual: "Frog",
      correct: true,
    },
    {
      id: 8,
      image: "/horse.jpg",
      predicted: "Cat",
      confidence: 73.4,
      actual: "Horse",
      correct: false,
    },
  ]

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Recent Inference Results</h2>
          <p className="text-sm text-muted-foreground mt-1">Latest predictions from the model</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className="font-mono">
            Batch Size: 32
          </Badge>
          <Badge variant="secondary" className="font-mono">
            12.3ms avg
          </Badge>
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {predictions.map((prediction) => (
          <div
            key={prediction.id}
            className="group relative rounded-lg border border-border bg-card p-4 hover:border-primary/50 transition-colors"
          >
            <div className="aspect-square rounded-md bg-muted mb-3 overflow-hidden">
              <Image
                src={prediction.image || "/placeholder.svg"}
                alt={prediction.predicted}
                width={80}
                height={80}
                className="w-full h-full object-cover"
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">{prediction.predicted}</span>
                {prediction.correct ? (
                  <Badge variant="default" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">
                    Correct
                  </Badge>
                ) : (
                  <Badge variant="default" className="bg-red-500/10 text-red-500 border-red-500/20">
                    Wrong
                  </Badge>
                )}
              </div>

              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-mono font-medium text-foreground">{prediction.confidence}%</span>
              </div>

              {!prediction.correct && (
                <div className="pt-2 border-t border-border">
                  <span className="text-xs text-muted-foreground">
                    Actual: <span className="text-foreground font-medium">{prediction.actual}</span>
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-border">
        <div>
          <p className="text-xs text-muted-foreground">Total Predictions</p>
          <p className="text-xl font-semibold text-foreground mt-1">10,000</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Correct</p>
          <p className="text-xl font-semibold text-emerald-500 mt-1">9,240</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Accuracy</p>
          <p className="text-xl font-semibold text-foreground mt-1">92.4%</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground">Avg Confidence</p>
          <p className="text-xl font-semibold text-foreground mt-1">91.8%</p>
        </div>
      </div>
    </Card>
  )
}
