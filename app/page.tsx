"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
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
} from "lucide-react"

// Mock data for demo - in production this would come from the Python backend
const mockPositions = [
  { id: 1, symbol: "BTC-USD", side: "BUY", qty: 0.01, entry_price: 67500, current_price: 68200, pnl: 7.00, pnl_pct: 1.04, opened_at: "2h ago" },
  { id: 2, symbol: "ETH-USD", side: "BUY", qty: 0.5, entry_price: 3450, current_price: 3420, pnl: -15.00, pnl_pct: -0.87, opened_at: "4h ago" },
  { id: 3, symbol: "SOL-USD", side: "BUY", qty: 10, entry_price: 145.50, current_price: 148.20, pnl: 27.00, pnl_pct: 1.86, opened_at: "1h ago" },
]

const mockClosedTrades = [
  { id: 101, symbol: "BTC-USD", side: "BUY", entry_price: 66800, exit_price: 67200, pnl: 4.00, pnl_pct: 0.60, closed_at: "1h ago", exit_reason: "take_profit" },
  { id: 102, symbol: "ETH-USD", side: "BUY", entry_price: 3500, exit_price: 3480, pnl: -10.00, pnl_pct: -0.57, closed_at: "3h ago", exit_reason: "stop_loss" },
  { id: 103, symbol: "SOL-USD", side: "BUY", entry_price: 142.00, exit_price: 145.00, pnl: 30.00, pnl_pct: 2.11, closed_at: "5h ago", exit_reason: "manual" },
]

export default function Dashboard() {
  const [positions, setPositions] = useState(mockPositions)
  const [closedTrades, setClosedTrades] = useState(mockClosedTrades)
  const [botStatus, setBotStatus] = useState({ enabled: true, dryRun: false, running: true })
  const [loading, setLoading] = useState(false)

  const totalUnrealized = positions.reduce((sum, p) => sum + p.pnl, 0)
  const totalRealized = closedTrades.reduce((sum, t) => sum + t.pnl, 0)
  const winRate = (closedTrades.filter(t => t.pnl > 0).length / closedTrades.length * 100).toFixed(1)

  const handleRefresh = () => {
    setLoading(true)
    // Simulate refresh
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
      <header className="border-b bg-card">
        <div className="container flex h-16 items-center justify-between px-4">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold">AlphaPilot AI</h1>
            <Badge variant={botStatus.running ? "default" : "secondary"}>
              {botStatus.running ? "Running" : "Stopped"}
            </Badge>
            {botStatus.dryRun && <Badge variant="outline">Dry Run</Badge>}
          </div>
          <nav className="flex items-center gap-2">
            <Link href="/">
              <Button variant="ghost" size="sm">Dashboard</Button>
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
        <div className="grid gap-4 md:grid-cols-4 mb-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Unrealized P&L</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${totalUnrealized >= 0 ? "text-green-500" : "text-red-500"}`}>
                {totalUnrealized >= 0 ? "+" : ""}{totalUnrealized.toFixed(2)} USD
              </div>
              <p className="text-xs text-muted-foreground">{positions.length} open positions</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Realized P&L</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${totalRealized >= 0 ? "text-green-500" : "text-red-500"}`}>
                {totalRealized >= 0 ? "+" : ""}{totalRealized.toFixed(2)} USD
              </div>
              <p className="text-xs text-muted-foreground">{closedTrades.length} closed trades</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{winRate}%</div>
              <p className="text-xs text-muted-foreground">
                {closedTrades.filter(t => t.pnl > 0).length}W / {closedTrades.filter(t => t.pnl <= 0).length}L
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Bot Status</CardTitle>
              <Wallet className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-2">
                {botStatus.running ? (
                  <Badge className="bg-green-500">Active</Badge>
                ) : (
                  <Badge variant="secondary">Stopped</Badge>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {botStatus.dryRun ? "Dry run mode" : "Live trading"}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Active Positions */}
        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Active Positions</CardTitle>
                <CardDescription>Manage open positions in real-time</CardDescription>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading}>
                  <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
                  Refresh
                </Button>
                <Button variant="outline" size="sm" onClick={handleTakeProfits}>
                  <TrendingUp className="h-4 w-4 mr-1" />
                  Take Profits
                </Button>
                <Button variant="outline" size="sm" onClick={handleCloseAll}>
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
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Symbol</th>
                      <th className="text-left py-2 px-2">Side</th>
                      <th className="text-right py-2 px-2">Qty</th>
                      <th className="text-right py-2 px-2">Entry</th>
                      <th className="text-right py-2 px-2">Current</th>
                      <th className="text-right py-2 px-2">P&L</th>
                      <th className="text-right py-2 px-2">P&L %</th>
                      <th className="text-left py-2 px-2">Opened</th>
                      <th className="text-left py-2 px-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((p) => (
                      <tr key={p.id} className="border-b hover:bg-muted/50">
                        <td className="py-2 px-2 font-medium">{p.symbol}</td>
                        <td className="py-2 px-2">
                          <Badge variant={p.side === "BUY" ? "default" : "destructive"}>
                            {p.side}
                          </Badge>
                        </td>
                        <td className="py-2 px-2 text-right font-mono">{p.qty}</td>
                        <td className="py-2 px-2 text-right font-mono">${p.entry_price.toLocaleString()}</td>
                        <td className="py-2 px-2 text-right font-mono">${p.current_price.toLocaleString()}</td>
                        <td className={`py-2 px-2 text-right font-mono ${p.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {p.pnl >= 0 ? "+" : ""}{p.pnl.toFixed(2)}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${p.pnl_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {p.pnl_pct >= 0 ? "+" : ""}{p.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="py-2 px-2 text-muted-foreground">{p.opened_at}</td>
                        <td className="py-2 px-2">
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => handleClosePosition(p.id)}
                          >
                            Close
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No open positions
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Trades */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Trades</CardTitle>
            <CardDescription>Last {closedTrades.length} closed trades</CardDescription>
          </CardHeader>
          <CardContent>
            {closedTrades.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-2">Symbol</th>
                      <th className="text-left py-2 px-2">Side</th>
                      <th className="text-right py-2 px-2">Entry</th>
                      <th className="text-right py-2 px-2">Exit</th>
                      <th className="text-right py-2 px-2">P&L</th>
                      <th className="text-right py-2 px-2">P&L %</th>
                      <th className="text-left py-2 px-2">Closed</th>
                      <th className="text-left py-2 px-2">Exit Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {closedTrades.map((t) => (
                      <tr key={t.id} className="border-b hover:bg-muted/50">
                        <td className="py-2 px-2 font-medium">{t.symbol}</td>
                        <td className="py-2 px-2">
                          <Badge variant={t.side === "BUY" ? "default" : "destructive"}>
                            {t.side}
                          </Badge>
                        </td>
                        <td className="py-2 px-2 text-right font-mono">${t.entry_price.toLocaleString()}</td>
                        <td className="py-2 px-2 text-right font-mono">${t.exit_price.toLocaleString()}</td>
                        <td className={`py-2 px-2 text-right font-mono ${t.pnl >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {t.pnl >= 0 ? "+" : ""}{t.pnl.toFixed(2)}
                        </td>
                        <td className={`py-2 px-2 text-right font-mono ${t.pnl_pct >= 0 ? "text-green-500" : "text-red-500"}`}>
                          {t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%
                        </td>
                        <td className="py-2 px-2 text-muted-foreground">{t.closed_at}</td>
                        <td className="py-2 px-2">
                          <Badge variant="outline">{t.exit_reason}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                No closed trades yet
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
