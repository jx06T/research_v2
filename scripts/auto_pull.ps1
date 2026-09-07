<#
.SYNOPSIS
    自動從遠端 Colab (透過 SSH/Cloudflare Tunnel) 拉取最新的模型權重至本地端。
.DESCRIPTION
    本腳本會在背景定時輪詢遠端目錄，自動將 G_latest.pth 與最新產生的 G_epoch_*.pth 下載到本地 runs/ 資料夾。
.EXAMPLE
    # 使用預設值啟動監控
    .\scripts\auto_pull.ps1

    # 自訂遠端主機名與目錄
    .\scripts\auto_pull.ps1 -RemoteHost "colab" -RemoteDir "/content/drive/MyDrive/research_v2_runs/exp_01" -LocalDir "runs/exp_01" -IntervalSeconds 60
#>

param(
    [string]$RemoteHost = "colab",
    [string]$RemoteDir = "/content/drive/MyDrive/research_v2_runs/exp_01",
    [string]$LocalDir = "runs/exp_01",
    [int]$IntervalSeconds = 60
)

# 確保本地輸出目錄存在
if (!(Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null
    Write-Host "[初始化] 已建立本地接收資料夾: $LocalDir" -ForegroundColor Cyan
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Colab 模型權重自動同步腳本 (Auto-Pull Daemon) 啟動中..." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "遠端主機 (Host): $RemoteHost" -ForegroundColor Yellow
Write-Host "遠端路徑 (Remote): $RemoteDir/*.pth" -ForegroundColor Yellow
Write-Host "本地路徑 (Local):  $LocalDir" -ForegroundColor Yellow
Write-Host "輪詢間隔 (Interval): 每 $IntervalSeconds 秒一次" -ForegroundColor Yellow
Write-Host "按 Ctrl + C 可隨時停止腳本。" -ForegroundColor DarkGray
Write-Host "----------------------------------------------------------"

$cycle = 1

while ($true) {
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    # 執行 scp 下載
    # 注意：使用 2>$null 忽略未找到新檔案時的 stderr 警告
    scp "${RemoteHost}:${RemoteDir}/*.pth" "$LocalDir" 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        $files = Get-ChildItem -Path $LocalDir -Filter "*.pth" | Sort-Object LastWriteTime -Descending
        $fileCount = $files.Count
        $latestFile = if ($fileCount -gt 0) { $files[0].Name } else { "無" }
        Write-Host "[$timestamp] [輪次 #$cycle] 同步成功！本地現有 $fileCount 個權重檔 (最新: $latestFile)" -ForegroundColor Green
    } else {
        Write-Host "[$timestamp] [輪次 #$cycle] 檢查中... (遠端尚無新權重或連線等待中)" -ForegroundColor DarkGray
    }

    $cycle++
    Start-Sleep -Seconds $IntervalSeconds
}

