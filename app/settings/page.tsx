"use client"

import { useState } from "react"
import Link from "next/link"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ArrowLeft, Settings, Save, AlertTriangle } from "lucide-react"

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    botEnabled: true,
    dryRun: false,
    tradingStyle: "hybrid",
    positionSizeUsd: 10,
    maxOpenPositions: 5,
    minConfidence: 0.55,
    tickSeconds: 120,
    microProfitTarget: 0.25,
    minProfitPct: 0.3,
    autoReinvest: true,
    maxDailyTrades: 100,
    stopLossPct: 10,
    takeProfitPct: 5,
    trailingStopPct: 3,
  })

  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    // In production this would call the API
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
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
            <h1 className="text-xl font-bold">Settings</h1>
          </div>
          <nav className="flex items-center gap-2">
            <Link href="/">
              <Button variant="ghost" size="sm">Dashboard</Button>
            </Link>
            <Link href="/analytics">
              <Button variant="ghost" size="sm">Analytics</Button>
            </Link>
            <Link href="/settings">
              <Button variant="ghost" size="sm" className="bg-accent">
                <Settings className="h-4 w-4" />
              </Button>
            </Link>
          </nav>
        </div>
      </header>

      <main className="container px-4 py-6 max-w-4xl">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold">Bot Settings</h2>
            <p className="text-muted-foreground">Configure your trading bot behavior</p>
          </div>
          <Button onClick={handleSave}>
            <Save className="h-4 w-4 mr-2" />
            {saved ? "Saved!" : "Save Changes"}
          </Button>
        </div>

        <div className="space-y-6">
          {/* Bot Status */}
          <Card>
            <CardHeader>
              <CardTitle>Bot Status</CardTitle>
              <CardDescription>Control the trading bot</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="botEnabled">Bot Enabled</Label>
                  <p className="text-sm text-muted-foreground">Enable or disable the trading bot</p>
                </div>
                <Switch
                  id="botEnabled"
                  checked={settings.botEnabled}
                  onCheckedChange={(checked) => setSettings({ ...settings, botEnabled: checked })}
                />
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="dryRun" className="flex items-center gap-2">
                    Dry Run Mode
                    {settings.dryRun && <Badge variant="outline">Active</Badge>}
                  </Label>
                  <p className="text-sm text-muted-foreground">Log decisions without executing trades</p>
                </div>
                <Switch
                  id="dryRun"
                  checked={settings.dryRun}
                  onCheckedChange={(checked) => setSettings({ ...settings, dryRun: checked })}
                />
              </div>
            </CardContent>
          </Card>

          {/* Trading Style */}
          <Card>
            <CardHeader>
              <CardTitle>Trading Style</CardTitle>
              <CardDescription>Configure how the bot trades</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Trading Style</Label>
                <Select
                  value={settings.tradingStyle}
                  onValueChange={(value) => setSettings({ ...settings, tradingStyle: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="scalper">Scalper - Quick micro profits ($0.25+)</SelectItem>
                    <SelectItem value="hybrid">Hybrid - Balance speed and size</SelectItem>
                    <SelectItem value="swing">Swing - Hold for bigger moves</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {settings.tradingStyle === "scalper" && (
                <div className="space-y-2">
                  <Label>Micro Profit Target (USD)</Label>
                  <Input
                    type="number"
                    step="0.05"
                    value={settings.microProfitTarget}
                    onChange={(e) => setSettings({ ...settings, microProfitTarget: parseFloat(e.target.value) })}
                  />
                  <p className="text-xs text-muted-foreground">Exit when profit reaches this amount</p>
                </div>
              )}

              <div className="space-y-2">
                <Label>Min Profit % to Take</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[settings.minProfitPct]}
                    onValueChange={([value]) => setSettings({ ...settings, minProfitPct: value })}
                    min={0.1}
                    max={5}
                    step={0.1}
                    className="flex-1"
                  />
                  <span className="w-16 text-right font-mono">{settings.minProfitPct}%</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="autoReinvest">Auto Reinvest Profits</Label>
                  <p className="text-sm text-muted-foreground">Automatically compound gains</p>
                </div>
                <Switch
                  id="autoReinvest"
                  checked={settings.autoReinvest}
                  onCheckedChange={(checked) => setSettings({ ...settings, autoReinvest: checked })}
                />
              </div>
            </CardContent>
          </Card>

          {/* Position Sizing */}
          <Card>
            <CardHeader>
              <CardTitle>Position Sizing</CardTitle>
              <CardDescription>Control trade sizes and limits</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Position Size (USD)</Label>
                  <Input
                    type="number"
                    value={settings.positionSizeUsd}
                    onChange={(e) => setSettings({ ...settings, positionSizeUsd: parseFloat(e.target.value) })}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max Open Positions</Label>
                  <Input
                    type="number"
                    value={settings.maxOpenPositions}
                    onChange={(e) => setSettings({ ...settings, maxOpenPositions: parseInt(e.target.value) })}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Max Daily Trades</Label>
                  <Input
                    type="number"
                    value={settings.maxDailyTrades}
                    onChange={(e) => setSettings({ ...settings, maxDailyTrades: parseInt(e.target.value) })}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Tick Interval (seconds)</Label>
                  <Input
                    type="number"
                    value={settings.tickSeconds}
                    onChange={(e) => setSettings({ ...settings, tickSeconds: parseInt(e.target.value) })}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Risk Management */}
          <Card>
            <CardHeader>
              <CardTitle>Risk Management</CardTitle>
              <CardDescription>Stop loss and take profit settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Min Confidence Threshold</Label>
                <div className="flex items-center gap-4">
                  <Slider
                    value={[settings.minConfidence * 100]}
                    onValueChange={([value]) => setSettings({ ...settings, minConfidence: value / 100 })}
                    min={30}
                    max={90}
                    step={5}
                    className="flex-1"
                  />
                  <span className="w-16 text-right font-mono">{(settings.minConfidence * 100).toFixed(0)}%</span>
                </div>
                <p className="text-xs text-muted-foreground">Minimum signal confidence to enter a trade</p>
              </div>

              <Separator />

              <div className="grid grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>Stop Loss %</Label>
                  <Input
                    type="number"
                    value={settings.stopLossPct}
                    onChange={(e) => setSettings({ ...settings, stopLossPct: parseFloat(e.target.value) })}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Take Profit %</Label>
                  <Input
                    type="number"
                    value={settings.takeProfitPct}
                    onChange={(e) => setSettings({ ...settings, takeProfitPct: parseFloat(e.target.value) })}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Trailing Stop %</Label>
                  <Input
                    type="number"
                    value={settings.trailingStopPct}
                    onChange={(e) => setSettings({ ...settings, trailingStopPct: parseFloat(e.target.value) })}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Danger Zone */}
          <Card className="border-destructive/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-destructive">
                <AlertTriangle className="h-5 w-5" />
                Danger Zone
              </CardTitle>
              <CardDescription>Irreversible actions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Emergency Kill Switch</p>
                  <p className="text-sm text-muted-foreground">Close all positions and halt trading</p>
                </div>
                <Button variant="destructive">Engage Kill Switch</Button>
              </div>
              <Separator />
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Reset Paper Balance</p>
                  <p className="text-sm text-muted-foreground">Reset all wallets to starting balance</p>
                </div>
                <Button variant="outline">Reset Balances</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
