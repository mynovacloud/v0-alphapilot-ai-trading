"use client"

import { useState, useEffect, useMemo } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import {
  Area,
  AreaChart,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts"
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Wallet,
  BarChart3,
  Settings,
  RefreshCw,
  AlertTriangle,
  PlayCircle,
  StopCircle,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  Target,
  Shield,
  Zap,
} from "lucide-react"

// Mock data for demo - in production this would come from the Python backend
const mockPositions = [
  { id: 1, symbol: "BTC-USD", side: "BUY", qty: 0.01, entry_price: 67500, current_price: 68200, pnl: 7.00, pnl_pct: 1.04, opened_at: "2h ago", stop_loss: 66500, take_profit: 69500 },
  { id: 2, symbol: "ETH-USD", side: "BUY", qty: 0.5, entry_price: 3450, current_price: 3420, pnl: -15.00, pnl_pct: -0.87, opened_at: "4h ago", stop_loss: 3350, take_profit: 3600 },
  { id: 3, symbol: "SOL-USD", side: "BUY", qty: 10, entry_price: 145.50, current_price: 148.20, pnl: 27.00, pnl_pct: 1.86, opened_at: "1h ago", stop_loss: 141, take_profit: 155 },
]

const mockClosedTrades = [
  { id: 101, symbol: "BTC-USD", side: "BUY", entry_price: 66800, exit_price: 67200, pnl: 4.00, pnl_pct: 0.60, closed_at: "1h ago", exit_reason: "take_profit" },
  { id: 102, symbol: "ETH-USD", side: "BUY", entry_price: 3500, exit_price: 3480, pnl: -10.00, pnl_pct: -0.57, closed_at: "3h ago", exit_reason: "stop_loss" },
  { id: 103, symbol: "SOL-USD", side: "BUY", entry_price: 142.00, exit_price: 145.00, pnl: 30.00, pnl_pct: 2.11, closed_at: "5h ago", exit_reason: "manual" },
]

// P&L history data for the chart (last 24 hours)
const mockPnlHistory = [
  { time: "00:00", pnl: 0, cumulative: 0 },
  { time: "02:00", pnl: 12, cumulative: 12 },
  { time: "04:00", pnl: -5, cumulative: 7 },
  { time: "06:00", pnl: 8, cumulative: 15 },
  { time: "08:00", pnl: 25, cumulative: 40 },
  { time: "10:00", pnl: -10, cumulative: 30 },
  { time: "12:00", pnl: 15, cumulative: 45 },
  { time: "14:00", pnl: -8, cumulative: 37 },
  { time: "16:00", pnl: 20, cumulative: 57 },
  { time: "18:00", pnl: 5, cumulative: 62 },
  { time: "20:00", pnl: -12, cumulative: 50 },
  { time: "22:00", pnl: 18, cumulative: 68 },
  { time: "Now", pnl: 7, cumulative: 75 },
]

const chartConfig = {
  cumulative: {
    label: "P&L",
    color: "hsl(142 76% 36%)",
  },
}

export default function Dashboard() {
  const [positions, setPositions] = useState(mockPositions)
  const [closedTrades, setClosedTrades] = useState(mockClosedTrades)
  const [pnlHistory, setPnlHistory] = useState(mockPnlHistory)
  const [botStatus, setBotStatus] = useState({ enabled: true, dryRun: false, running: true })
  const [loading, setLoading] = useState(false)

  const totalUnrealized = positions.reduce((sum, p) => sum + p.pnl, 0)
  const totalRealized = closedTrades.reduce((sum, t) => sum + t.pnl, 0)
  const winCount = closedTrades.filter(t => t.pnl > 0).length
  const lossCount = closedTrades.filter(t => t.pnl <= 0).length
  const winRate = closedTrades.length > 0 ? (winCount / closedTrades.length * 100).toFixed(1) : "0.0"
  
  // Calculate additional metrics
  const avgWin = winCount > 0 
    ? closedTrades.filter(t => t.pnl > 0).reduce((sum, t) => sum + t.pnl, 0) / winCount 
    : 0
  const avgLoss = lossCount > 0 
    ? closedTrades.filter(t => t.pnl <= 0).reduce((sum, t) => sum + t.pnl, 0) / lossCount 
    : 0
  const profitFactor = Math.abs(avgLoss) > 0 ? Math.abs(avgWin / avgLoss) : 0

  const handleRefresh = () => {
    setLoading(true)
    setTimeout(() => setLoading(false), 500)
  }

  const handleClosePosition = (id: number) => {
    if (confirm("Close this position?")) {
      setPositions(positions.filter(p => p.id !== id))
    }
  }

  const handleCloseAll = () => {
    if (confirm("Close ALL positions?")) {
      setPositions([])
    }
  }

  const handleTakeProfits = () => {
    const profitable = positions.filter(p => p.pnl > 0)
    if (confirm(`Close ${profitable.length} profitable positions?`)) {
      setPositions(positions.filter(p => p.pnl <= 0))
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
                <Zap className="h-5 w-5 text-primary-foreground" />
              </div>
              <h1 className="text-xl font-bold">AlphaPilot AI</h1>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={botStatus.running ? "default" : "secondary"} className={botStatus.running ? "bg-green-500/10 text-green-500 hover:bg-green-500/20" : ""}>
                <span className={`mr-1.5 h-2 w-2 rounded-full ${botStatus.running ? "bg-green-500 animate-pulse" : "bg-muted-foreground"}`} />
                {botStatus.running ? "Running" : "Stopped"}
              </Badge>
              {botStatus.dryRun && <Badge variant="outline">Dry Run</Badge>}
            </div>
          </div>
          <nav className="flex items-center gap-2">
            <Link href="/">
              <Button variant="ghost" size="sm" className="bg-accent">Dashboard</Button>
            </Link>
            <Link href="/analytics">
              <Button variant="ghost" size="sm">Analytics</Button>
            </Link>
            <Link href="/settings">
              <Button variant="ghost" size="sm">
                <Settings className="h-4 w-4" />
              </Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="container px-4 py-6">
        {/* KPI Cards */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-6">
          <Card className="relative overflow-hidden">
            <div className={`absolute inset-0 opacity-10 ${totalUnrealized >= 0 ? "bg-gradient-to-br from-green-500" : "bg-gradient-to-br from-red-500"}`} />
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Unrealized P&L</CardTitle>
              {totalUnrealized >= 0 ? (
                <ArrowUpRight className="h-4 w-4 text-green-500" />
              ) : (
                <ArrowDownRight className="h-4 w-4 text-red-500" />
              )}
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${totalUnrealized >= 0 ? "text-green-500" : "text-red-500"}`}>
                {totalUnrealized >= 0 ? "+" : ""}{totalUnrealized.toFixed(2)} USD
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {positions.length} open position{positions.length !== 1 ? "s" : ""}
              </p>
            </CardContent>
          </Card>

          <Card className="relative overflow-hidden">
            <div className={`absolute inset-0 opacity-10 ${totalRealized >= 0 ? "bg-gradient-to-br from-green-500" : "bg-gradient-to-br from-red-500"}`} />
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Realized P&L</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${totalRealized >= 0 ? "text-green-500" : "text-red-500"}`}>
                {totalRealized >= 0 ? "+" : ""}{totalRealized.toFixed(2)} USD
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {closedTrades.length} closed trade{closedTrades.length !== 1 ? "s" : ""}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{winRate}%</div>
              <div className="flex items-center gap-2 mt-1">
                <Progress value={parseFloat(winRate)} className="h-1.5" />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                <span className="text-green-500">{winCount}W</span> / <span className="text-red-500">{lossCount}L</span>
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Profit Factor</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${profitFactor >= 1.5 ? "text-green-500" : profitFactor >= 1 ? "text-yellow-500" : "text-red-500"}`}>
                {profitFactor.toFixed(2)}x
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Avg Win ${avgWin.toFixed(2)} / Loss ${Math.abs(avgLoss).toFixed(2)}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* P&L Chart */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>P&L Performance</CardTitle>
                <CardDescription>Cumulative profit/loss over the last 24 hours</CardDescription>
              </div>
              <div className="text-right">
                <div className={`text-2xl font-bold ${pnlHistory[pnlHistory.length - 1].cumulative >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {pnlHistory[pnlHistory.length - 1].cumulative >= 0 ? "+" : ""}${pnlHistory[pnlHistory.length - 1].cumulative.toFixed(2)}
                </div>
                <p className="text-xs text-muted-foreground">Total P&L Today</p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-[200px] w-full">
              <AreaChart data={pnlHistory} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="fillPnl" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(142 76% 36%)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="hsl(142 76% 36%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis 
                  dataKey="time" 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fontSize: 12 }}
                  tickMargin={8}
                />
                <YAxis 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `$${value}`}
                  width={50}
                />
                <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                <ChartTooltip 
                  content={<ChartTooltipContent />}
                  formatter={(value) => [`$${value}`, "P&L"]}
                />
                <Area
                  type="monotone"
                  dataKey="cumulative"
                  stroke="hsl(142 76% 36%)"
                  strokeWidth={2}
                  fill="url(#fillPnl)"
                />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Active Positions */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  Active Positions
                  <Badge variant="secondary">{positions.length}</Badge>
                </CardTitle>
                <CardDescription>Manage open positions with real-time P&L</CardDescription>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading}>
                  <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
                <Button variant="outline" size="sm" onClick={handleTakeProfits} disabled={positions.filter(p => p.pnl > 0).length === 0}>
                  <TrendingUp className="h-4 w-4 mr-1" />
                  Take Profits
                </Button>
                <Button variant="outline" size="sm" onClick={handleCloseAll} disabled={positions.length === 0}>
                  Close All
                </Button>
                <Button variant="destructive" size="sm">
                  <AlertTriangle className="h-4 w-4 mr-1" />
                  Emergency Exit
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {positions.length > 0 ? (
              <div className="space-y-3">
                {positions.map((p) => (
                  <div key={p.id} className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-muted/50 transition-colors">
                    <div className="flex items-center gap-4">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-semibold">{p.symbol}</span>
                          <Badge variant={p.side === "BUY" ? "default" : "destructive"} className="text-xs">
                            {p.side}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {p.qty} @ ${p.entry_price.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground">Current</div>
                        <div className="font-mono">${p.current_price.toLocaleString()}</div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm text-muted-foreground">SL / TP</div>
                        <div className="font-mono text-xs">
                          <span className="text-red-500">${p.stop_loss}</span>
                          {" / "}
                          <span className="text-green-500">${p.take_profit}</span>
                        </div>
                      </div>
                      
                      <div className="text-right min-w-[100px]">
                        <div className={`text-lg font-bold ${p.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {p.pnl >= 0 ? "+" : ""}{p.pnl.toFixed(2)}
                        </div>
                        <div className={`text-sm ${p.pnl_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {p.pnl_pct >= 0 ? "+" : ""}{p.pnl_pct.toFixed(2)}%
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        <span className="text-xs">{p.opened_at}</span>
                      </div>
                      
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleClosePosition(p.id)}
                      >
                        Close
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Shield className="h-12 w-12 mx-auto mb-4 opacity-20" />
                <p>No open positions</p>
                <p className="text-sm">The bot will open positions when signals meet your criteria</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Trades */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Recent Trades
              <Badge variant="secondary">{closedTrades.length}</Badge>
            </CardTitle>
            <CardDescription>Last {closedTrades.length} closed trades</CardDescription>
          </CardHeader>
          <CardContent>
            {closedTrades.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-3 px-2 font-medium">Symbol</th>
                      <th className="text-left py-3 px-2 font-medium">Side</th>
                      <th className="text-right py-3 px-2 font-medium">Entry</th>
                      <th className="text-right py-3 px-2 font-medium">Exit</th>
                      <th className="text-right py-3 px-2 font-medium">P&L</th>
                      <th className="text-right py-3 px-2 font-medium">Return</th>
                      <th className="text-left py-3 px-2 font-medium">Closed</th>
                      <th className="text-left py-3 px-2 font-medium">Exit Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {closedTrades.map((t) => (
                      <tr key={t.id} className="border-b hover:bg-muted/50 transition-colors">
                        <td className="py-3 px-2 font-medium">{t.symbol}</td>
                        <td className="py-3 px-2">
                          <Badge variant={t.side === "BUY" ? "default" : "destructive"} className="text-xs">
                            {t.side}
                          </Badge>
                        </td>
                        <td className="py-3 px-2 text-right font-mono">${t.entry_price.toLocaleString()}</td>
                        <td className="py-3 px-2 text-right font-mono">${t.exit_price.toLocaleString()}</td>
                        <td className={`py-3 px-2 text-right font-mono font-semibold ${t.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {t.pnl >= 0 ? "+" : ""}{t.pnl.toFixed(2)}
                        </td>
                        <td className={`py-3 px-2 text-right font-mono ${t.pnl_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="py-3 px-2 text-muted-foreground">{t.closed_at}</td>
                        <td className="py-3 px-2">
                          <Badge 
                            variant="outline" 
                            className={
                              t.exit_reason === "take_profit" ? "border-green-500 text-green-500" :
                              t.exit_reason === "stop_loss" ? "border-red-500 text-red-500" :
                              ""
                            }
                          >
                            {t.exit_reason.replace("_", " ")}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Activity className="h-12 w-12 mx-auto mb-4 opacity-20" />
                <p>No closed trades yet</p>
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
