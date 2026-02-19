# 上传项目到 GitHub Fuess 分支指南

## 当前状态
- 远程仓库：`https://github.com/Fuess123/DiffQ-Rec.git`
- 当前分支：`main`
- 有未提交的更改和未跟踪的文件

## 步骤

### 1. 添加并提交当前更改

首先，添加所有更改的文件：

```powershell
# 添加所有修改的文件
git add .

# 或者选择性添加文件
git add src/data/loading/components/iterators.py
git add src/data/loading/components/pre_processing.py
git add src/modules/clustering/vq_vae.py
git add configs/experiment/vqvae_*.yaml
```

然后提交更改：

```powershell
git commit -m "修复UnicodeDecodeError问题，添加VQ-VAE模块"
```

### 2. 创建或切换到 Fuess 分支

如果 Fuess 分支不存在，创建它：

```powershell
# 创建并切换到 Fuess 分支
git checkout -b Fuess
```

如果 Fuess 分支已存在，切换到它：

```powershell
# 切换到 Fuess 分支
git checkout Fuess
```

### 3. 将 Fuess 分支推送到远程

首次推送 Fuess 分支：

```powershell
# 推送并设置上游分支
git push -u origin Fuess
```

之后只需要：

```powershell
git push
```

### 4. 如果需要在 Fuess 分支上合并 main 的更改

如果你想将 main 分支的更改合并到 Fuess：

```powershell
# 切换到 Fuess 分支
git checkout Fuess

# 合并 main 分支的更改
git merge main

# 解决冲突（如果有）后推送
git push
```

## 完整命令序列（推荐）

```powershell
# 1. 添加所有更改
git add .

# 2. 提交更改
git commit -m "修复UnicodeDecodeError问题，添加VQ-VAE模块和相关配置"

# 3. 创建并切换到 Fuess 分支
git checkout -b Fuess

# 4. 推送 Fuess 分支到远程
git push -u origin Fuess
```

## 注意事项

1. **日志文件**：`logs/` 目录下的文件通常不应该提交到 Git。检查 `.gitignore` 是否已忽略这些文件。

2. **大文件**：如果文件很大，考虑使用 Git LFS。

3. **敏感信息**：确保没有提交敏感信息（API密钥、密码等）。

4. **冲突处理**：如果远程 Fuess 分支已有内容，可能需要先拉取：
   ```powershell
   git pull origin Fuess
   ```

## 如果遇到问题

### 问题：推送被拒绝
```powershell
# 先拉取远程更改
git pull origin Fuess --rebase

# 然后推送
git push
```

### 问题：需要强制推送（谨慎使用）
```powershell
git push -f origin Fuess
```

**警告**：强制推送会覆盖远程分支的历史，请确保这是你想要的操作。

