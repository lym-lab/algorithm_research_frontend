# 发布个人 GitHub 公开源码仓库

只发布 `algorithm_frontend` 目录中的内容，使 `package.json` 位于仓库根目录。不要发布整个原项目父目录。

## 首次发布

1. 登录自己的 GitHub，新建空仓库 `algorithm_research_frontend`，账号使用 `lym-lab`，可见性选 **Public**。复制它的 HTTPS 仓库地址。不要把账号密码发给别人。
2. 在本机终端进入 `语音识别课/algorithm_frontend`。先检查目录只有前端源码、配置、获准公开的样例和说明。
3. 在此目录依次执行（如 Git 提示身份未配置，用自己的 GitHub 名称与邮箱配置）：

```bash
git init
git add .
git commit -m "Add algorithm frontend V1"
```

4. 创建成功后使用下面的仓库地址；如果你更改了名称，请同步替换：

```bash
git branch -M main
git remote add origin https://github.com/lym-lab/algorithm_research_frontend.git
git push -u origin main
```

使用 Git 的正常登录流程完成验证；不要在 URL 或源码中写密码、Token。也可以用 GitHub Desktop 选择这个目录并发布，取消私有仓库选项。

## 其他人拉取

```bash
git clone https://github.com/lym-lab/algorithm_research_frontend.git
cd algorithm_research_frontend
npm install
npm run dev
```

每组增加方法配置与样例，发起 Pull Request，审核后合并。同一仓库维护所有方法，不为每组新建前端仓库或注册域名。

## 源码共享和在线站点

公开仓库用于拉取源码，不自动提供在线算法服务。如果后续要一个公开网页，可以另行启用静态托管并发布 `npm run build` 的 `dist/`。前端使用相对资源路径与哈希路由，兼容仓库子路径；CPU / GPU 算法仍需可访问的独立 HTTP 服务。

本次按要求只整理 GitHub 源码，不继续部署实验室服务器。真正发布后，以 GitHub 返回的仓库地址为准。
