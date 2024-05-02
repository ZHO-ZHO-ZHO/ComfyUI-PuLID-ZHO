
![Dingtalk_20240502003757](https://github.com/ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO/assets/140084057/9cbfacde-6ea3-4bde-8d12-cefb42a49bcd)

# ComfyUI PuLID（WIP）

Unofficial implementation of [PuLID](https://github.com/ToTheBeginning/PuLID)（diffusers） for ComfyUI



## 项目介绍 | Info

- 来自对 [PuLID](https://github.com/ToTheBeginning/PuLID) 的非官方实现，是基于 diffusers 的简版插件，并非原生版
  
- 版本：V0.9，测试初版，已经可用（会自动下载所有模型）,支持单图/多图，还在持续优化中！别急！
 


## 使用示例

- 多风格迁移
  ![Dingtalk_20240502005308](https://github.com/ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO/assets/140084057/f802d24e-66a6-41dc-b4c5-254e520cdb8a)

- 多 ID 混合
  ![Dingtalk_20240502010755](https://github.com/ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO/assets/140084057/361c8a49-c4e9-4636-83d6-7641fc3b4c53)

- extremely style 模式：特征保持更好
  ![Dingtalk_20240502020011](https://github.com/ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO/assets/140084057/3d716cbe-6eb7-4e7d-8716-70f7f7683a1a)



## 节点说明 | Features

- 🪐PuLID
    - face_image：接入参考图
    - supp_images：接入多张参考图 / 混合ID（非必要项）
    - positivet、negative：正负提示词
    - width、height：尺寸设置（需 64 倍数）
    - id_scale：ID 强度
    - mode：fidelity（正常） 和 extremely style（更相似）
    - id_mix：是否混合 ID
    - step：步数，默认 4 步（采用了 Lightning 4 步模型）
    - cfg：提示词相关度，默认为 1.2
    - seed：种子



## 安装 | Install

- 推荐使用管理器 ComfyUI Manager 安装（ON THE WAY）

- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO.git`
    3. `cd custom_nodes/ComfyUI-PuLID-ZHO`
    4. `pip install -r requirements.txt`
    5. 重启 ComfyUI
 


## 更新日志

- 20240502

  初版上线

  创建项目



## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/ComfyUI-PuLID-ZHO&Date)



## 关于我 | About me

📬 **联系我**：
- 邮箱：zhozho3965@gmail.com
- QQ 群：839821928

🔗 **社交媒体**：
- 个人页：[-Zho-](https://jike.city/zho)
- Bilibili：[我的B站主页](https://space.bilibili.com/484366804)
- X（Twitter）：[我的Twitter](https://twitter.com/ZHOZHO672070)
- 小红书：[我的小红书主页](https://www.xiaohongshu.com/user/profile/63f11530000000001001e0c8?xhsshare=CopyLink&appuid=63f11530000000001001e0c8&apptime=1690528872)

💡 **支持我**：
- B站：[B站充电](https://space.bilibili.com/484366804)
- 爱发电：[为我充电](https://afdian.net/a/ZHOZHO)



## Credits

[PuLID](https://github.com/ToTheBeginning/PuLID)
