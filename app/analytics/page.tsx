"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Wallet,
  BarChart3,
  Settings,
  RefreshCw,
  ArrowLeft,
} from "lucide-react"

// Mock data for demo - in production this would come from the Python backend
const mockOpenPositions = [
  { id: 1, symbol: "BTC-USD", side: "BUY", qty: 0.01, entry_price: 67500, current_price: 68200, unrealized_pnl: 7.00, pnl_pct: 1.04, opened_at: "2026-05-18T01:00:00Z", duration_hours: 2.5 },
  { id: 2, symbol: "ETH-USD", side: "BUY", qty: 0.5, entry_price: 3450, current_price: 3420, unrealized_pnl: -15.00, pnl_pct: -0.87, opened_at: "2026-05-18T00:00:00Z", duration_hours: 4.2 },
  { id: 3, symbol: "SOL-USD", side: "BUY", qty: 10, entry_price: 145.50, current_price: 148.20, unrealized_pnl: 27.00, pnl_pct: 1.86, opened_at: "2026-05-18T02:30:00Z", duration_hours: 1.1 },
]

const mockClosedTrades = [
  { id: 101, symbol: "BTC-USD", side: "BUY", qty: 0.01, entry_price: 66800, exit_price: 67200, realized_pnl: 4.00, pnl_pct: 0.60, opened_at: "2026-05-17T20:00:00Z", closed_at: "2026-05-17T22:00:00Z", duration_hours: 2.0, exit_reason: "take_profit" },
  { id: 102, symbol: "ETH-USD", side: "BUY", qty: 0.3, entry_price: 3500, exit_price: 3480, realized_pnl: -6.00, pnl_pct: -0.57, opened_at: "2026-05-17T18:00:00Z", closed_at: "2026-05-17T20:00:00Z", duration_hours: 2.0, exit_reason: "stop_loss" },
  { id: 103, symbol: "SOL-USD", side: "BUY", qty: 20, entry_price: 142.00, exit_price: 145.00, realized_pnl: 60.00, pnl_pct: 2.11, opened_at: "2026-05-17T10:00:00Z", closed_at: "2026-05-17T15:00:00Z", duration_hours: 5.0, exit_reason: "manual" },
  { id: 104, symbol: "DOGE-USD", side: "BUY", qty: 1000, entry_price: 0.12, exit_price: 0.125, realized_pnl: 5.00, pnl_pct: 4.17, opened_at: "2026-05-17T08:00:00Z", closed_at: "2026-05-17T12:00:00Z", duration_hours: 4.0, exit_reason: "take_profit" },
  { id: 105, symbol: "BTC-USD", side: "BUY", qty: 0.005, entry_price: 67000, exit_price: 66500, realized_pnl: -2.50, pnl_pct: -0.75, opened_at: "2026-05-16T22:00:00Z", closed_at: "2026-05-17T02:00:00Z", duration_hours: 4.0, exit_reason: "stop_loss" },
  { id: 106, symbol: "ETH-USD", side: "BUY", qty: 0.2, entry_price: 3400, exit_price: 3450, realized_pnl: 10.00, pnl_pct: 1.47, opened_at: "2026-05-16T14:00:00Z", closed_at: "2026-05-16T20:00:00Z", duration_hours: 6.0, exit_reason: "take_profit" },
]

const mockSummary = {
  total_pnl: 70.50,
  unrealized_pnl: 19.00,
  daily_pnl: 24.00,
  weekly_pnl: 70.50,
  monthly_pnl: 70.50,
  ytd_pnl: 70.50,
  win_rate: 0.67,
}

const mockPerf = {
  profit_factor: 2.35,
  sharpe_placeholder: 1.82,
  max_drawdown: 0.032,
  avg_rr: 1.8,
  max_consecutive_wins: 3,
  max_consecutive_losses: 1,
  biggest_win: 60.00,
  biggest_loss: -6.00,
  avg_trade_duration_hours: 3.8,
  avg_win: 19.75,
  avg_loss: -4.25,
  total_closed: 6,
}

export default function AnalyticsPage() {
  const [openPositions, setOpenPositions] = useState(mockOpenPositions)
  const [closedTrades, setClosedTrades] = useState(mockClosedTrades)
  const [summary, setSummary] = useState(mockSummary)
  const [perf, setPerf] = useState(mockPerf)
  const [filterSymbol, setFilterSymbol] = useState("all")
  const [filterOutcome, setFilterOutcome] = useState("all")
  const [loading, setLoading] = useState(false)

  const filteredTrades = closedTrades.filter(t => {
    if (filterSymbol !== "all" && t.symbol !== filterSymbol) return false
    if (filterOutcome === "win" && t.realized_pnl <= 0) return false
    if (filterOutcome === "loss" && t.realized_pnl > 0) return false
    return true
  })

  const symbols = [...new Set([...openPositions.map(p => p.symbol), ...closedTrades.map(t => t.symbol)])]
  const totalTrades = openPositions.length + closedTrades.length

  const handleRefresh = () => {
    setLoading(true)
    setTimeout(() => setLoading(false), 500)
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="h-4 w-4 mr-1" />
                Back
              </Button>
            </Link>
            <h1 className="text-xl font-bold">Analytics</h1>
          </div>
          <nav className="flex items-center gap-2">
            <Link href="/">
              <Button variant="ghost" size="sm">Dashboard</Button>
            </Link>
            <Link href="/analytics">
              <Button variant="ghost" size="sm" className="bg-accent">Analytics</Button>
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
        {/* Page Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold">Analytics</h2>
            <p className="text-muted-foreground">Complete trading history and performance breakdown</p>
          </div>
          <Button variant="outline" onClick={handleRefresh} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>

        {/* Summary KPIs */}
        <div className="grid gap-4 md:grid-cols-4 lg:grid-cols-8 mb-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Total Trades</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{totalTrades}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Win Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${summary.win_rate >= 0.5 ? "text-green-500" : "text-red-500"}`}>
                {(summary.win_rate * 100).toFixed(1)}%
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Profit Factor</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{perf.profit_factor.toFixed(2)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Total P&L</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${summary.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                {summary.total_pnl >= 0 ? "+" : ""}${summary.total_pnl.toFixed(2)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Unrealized P&L</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${summary.unrealized_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                {summary.unrealized_pnl >= 0 ? "+" : ""}${summary.unrealized_pnl.toFixed(2)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Open Positions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{openPositions.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Best Win</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-500">+${perf.biggest_win.toFixed(2)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Worst Loss</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-500">${perf.biggest_loss.toFixed(2)}</div>
            </CardContent>
          </Card>
        </div>

        {/* Period P&L and Advanced Metrics */}
        <div className="grid gap-4 md:grid-cols-2 mb-6">
          <Card>
            <CardHeader>
              <CardTitle>Period P&L</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Daily</span>
                <span className={`font-mono ${summary.daily_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {summary.daily_pnl >= 0 ? "+" : ""}${summary.daily_pnl.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Weekly</span>
                <span className={`font-mono ${summary.weekly_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {summary.weekly_pnl >= 0 ? "+" : ""}${summary.weekly_pnl.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Monthly</span>
                <span className={`font-mono ${summary.monthly_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {summary.monthly_pnl >= 0 ? "+" : ""}${summary.monthly_pnl.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">YTD</span>
                <span className={`font-mono ${summary.ytd_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {summary.ytd_pnl >= 0 ? "+" : ""}${summary.ytd_pnl.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between pt-2 border-t font-medium">
                <span className="text-muted-foreground">Total Realized</span>
                <span className={`font-mono ${summary.total_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                  {summary.total_pnl >= 0 ? "+" : ""}${summary.total_pnl.toFixed(2)}
                </span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Advanced Metrics</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Avg Win</span>
                <span className="font-mono text-green-500">${perf.avg_win.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Avg Loss</span>
                <span className="font-mono text-red-500">${perf.avg_loss.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Avg R:R</span>
                <span className="font-mono">{perf.avg_rr.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Max Drawdown</span>
                <span className="font-mono text-red-500">{(perf.max_drawdown * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Win Streak</span>
                <span className="font-mono text-green-500">{perf.max_consecutive_wins}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Loss Streak</span>
                <span className="font-mono text-red-500">{perf.max_consecutive_losses}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Avg Duration</span>
                <span className="font-mono">{perf.avg_trade_duration_hours.toFixed(1)}h</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground text-sm">Sharpe (proxy)</span>
                <span className="font-mono">{perf.sharpe_placeholder.toFixed(2)}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Open Positions */}
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Open Positions ({openPositions.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {openPositions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Symbol</th>
                      <th className="text-left py-2 px-2">Side</th>
                      <th className="text-right py-2 px-2">Qty</th>
                      <th className="text-right py-2 px-2">Entry</th>
                      <th className="text-right py-2 px-2">Current</th>
                      <th className="text-right py-2 px-2">Unrealized P&L</th>
                      <th className="text-right py-2 px-2">P&L %</th>
                      <th className="text-left py-2 px-2">Opened</th>
                      <th className="text-right py-2 px-2">Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {openPositions.map((p) => (
                      <tr key={p.id} className="border-b hover:bg-muted/50">
                        <td className="py-2 px-2 font-medium">{p.symbol}</td>
                        <td className="py-2 px-2">
                          <Badge variant={p.side === "BUY" ? "default" : "destructive"}>{p.side}</Badge>
                        </td>
                        <td className="py-2 px-2 text-right font-mono">{p.qty}</td>
                        <td className="py-2 px-2 text-right font-mono">${p.entry_price.toLocaleString()}</td>
                        <td className="py-2 px-2 text-right font-mono">${p.current_price.toLocaleString()}</td>
                        <td className={`py-2 px-2 text-right font-mono ${p.unrealized_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {p.unrealized_pnl >= 0 ? "+" : ""}${p.unrealized_pnl.toFixed(2)}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${p.pnl_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {p.pnl_pct >= 0 ? "+" : ""}{p.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="py-2 px-2 text-muted-foreground">{formatDate(p.opened_at)}</td>
                        <td className="py-2 px-2 text-right font-mono">{p.duration_hours.toFixed(1)}h</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">No open positions</div>
            )}
          </CardContent>
        </Card>

        {/* Trade History */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Trade History ({closedTrades.length} closed trades)</CardTitle>
              <div className="flex gap-2">
                <Select value={filterSymbol} onValueChange={setFilterSymbol}>
                  <SelectTrigger className="w-[130px]">
                    <SelectValue placeholder="All Symbols" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Symbols</SelectItem>
                    {symbols.map(sym => (
                      <SelectItem key={sym} value={sym}>{sym}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={filterOutcome} onValueChange={setFilterOutcome}>
                  <SelectTrigger className="w-[100px]">
                    <SelectValue placeholder="All" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="win">Winners</SelectItem>
                    <SelectItem value="loss">Losers</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {filteredTrades.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Symbol</th>
                      <th className="text-left py-2 px-2">Side</th>
                      <th className="text-right py-2 px-2">Qty</th>
                      <th className="text-right py-2 px-2">Entry</th>
                      <th className="text-right py-2 px-2">Exit</th>
                      <th className="text-right py-2 px-2">P&L</th>
                      <th className="text-right py-2 px-2">P&L %</th>
                      <th className="text-left py-2 px-2">Opened</th>
                      <th className="text-left py-2 px-2">Closed</th>
                      <th className="text-right py-2 px-2">Duration</th>
                      <th className="text-left py-2 px-2">Exit Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredTrades.map((t) => (
                      <tr key={t.id} className="border-b hover:bg-muted/50">
                        <td className="py-2 px-2 font-medium">{t.symbol}</td>
                        <td className="py-2 px-2">
                          <Badge variant={t.side === "BUY" ? "default" : "destructive"}>{t.side}</Badge>
                        </td>
                        <td className="py-2 px-2 text-right font-mono">{t.qty}</td>
                        <td className="py-2 px-2 text-right font-mono">${t.entry_price.toLocaleString()}</td>
                        <td className="py-2 px-2 text-right font-mono">${t.exit_price.toLocaleString()}</td>
                        <td className={`py-2 px-2 text-right font-mono ${t.realized_pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {t.realized_pnl >= 0 ? "+" : ""}${t.realized_pnl.toFixed(2)}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${t.pnl_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="py-2 px-2 text-muted-foreground text-xs">{formatDate(t.opened_at)}</td>
                        <td className="py-2 px-2 text-muted-foreground text-xs">{formatDate(t.closed_at)}</td>
                        <td className="py-2 px-2 text-right font-mono">{t.duration_hours.toFixed(1)}h</td>
                        <td className="py-2 px-2">
                          <Badge variant="outline">{t.exit_reason}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">No trades match the filter</div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
