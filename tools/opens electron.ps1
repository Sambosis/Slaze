try {
    $projectPath = "C:\Users\Machine81\Slazy\repo\electron"
    
    if (-Not (Test-Path $projectPath)) {
        throw "Project directory does not exist: $projectPath"
    }
    
    # Start a new command prompt and run the command.
    # The /K switch keeps the window open after executing the command.
    Start-Process cmd.exe -ArgumentList "/K cd `"$projectPath`" && npm start"
}
catch {
    Write-Error "Error starting npm: $_"
}