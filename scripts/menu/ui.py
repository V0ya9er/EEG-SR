#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI 组件模块

提供跨平台的终端 UI 组件，优先使用纯 ASCII 实现以确保兼容性。
"""

import os
import sys
import platform
import time
import threading
from typing import List, Optional, Dict, Any, Callable

# ============================================================================
# 终端控制
# ============================================================================

def clear_screen():
    """清屏"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def get_terminal_size() -> tuple:
    """获取终端尺寸"""
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        return 80, 24  # 默认尺寸


# ============================================================================
# 基础打印函数
# ============================================================================

def print_header(title: str = "SR-EEG 随机共振脑电分类实验", subtitle: str = "运行助手"):
    """打印应用标题头"""
    clear_screen()
    width = 64
    print()
    print("=" * width)
    print(f"{title:^{width}}")
    if subtitle:
        print(f"{subtitle:^{width}}")
    print("=" * width)
    print()


def print_menu_header(title: str, icon: str = ""):
    """打印菜单标题"""
    width = 60
    if icon:
        title = f"{icon} {title}"
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_separator(char: str = "-", width: int = 60):
    """打印分隔线"""
    print(char * width)


def print_success(message: str):
    """打印成功消息"""
    print(f"  [OK] {message}")


def print_error(message: str):
    """打印错误消息"""
    print(f"  [!] {message}")


def print_warning(message: str):
    """打印警告消息"""
    print(f"  [*] {message}")


def print_info(message: str):
    """打印信息消息"""
    print(f"  [i] {message}")


# ============================================================================
# 输入函数
# ============================================================================

def wait_for_enter(message: str = "按回车键继续..."):
    """等待用户按回车"""
    input(f"\n  {message}")


def get_input(prompt: str, default: str = "") -> str:
    """获取用户输入"""
    result = input(prompt).strip()
    return result if result else default


def get_choice(prompt: str, valid_choices: List[str], 
               case_sensitive: bool = False) -> str:
    """获取用户选择"""
    while True:
        choice = input(prompt).strip()
        if not case_sensitive:
            choice_check = choice.lower()
            valid_check = [v.lower() for v in valid_choices]
        else:
            choice_check = choice
            valid_check = valid_choices
            
        if choice_check in valid_check:
            return choice
        print(f"  无效选择，请输入 {'/'.join(valid_choices)}")


def confirm(prompt: str = "确认? [Y/n]: ", default: bool = True) -> bool:
    """确认操作"""
    choice = input(prompt).strip().lower()
    if not choice:
        return default
    return choice in ('y', 'yes', '是')


def get_validated_input(prompt: str, 
                        validator: Callable[[str], Any],
                        error_msg: str = "输入无效",
                        default: Any = None) -> Any:
    """带验证的输入获取"""
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        
        try:
            value = validator(raw)
            return value
        except (ValueError, TypeError):
            print(f"  [!] {error_msg}")


# ============================================================================
# Table 组件
# ============================================================================

class Table:
    """ASCII 表格组件"""
    
    def __init__(self, headers: List[str], 
                 col_widths: Optional[List[int]] = None,
                 padding: int = 2):
        """
        初始化表格
        
        Args:
            headers: 表头列表
            col_widths: 列宽列表，None 表示自动计算
            padding: 列之间的间距
        """
        self.headers = headers
        self.rows: List[List[str]] = []
        self.col_widths = col_widths
        self.padding = padding
        self.highlight_row: Optional[int] = None
    
    def add_row(self, row: List[Any]):
        """添加一行数据"""
        self.rows.append([str(cell) for cell in row])
    
    def set_highlight(self, row_index: int):
        """设置高亮行"""
        self.highlight_row = row_index
    
    def _calculate_widths(self) -> List[int]:
        """计算列宽"""
        if self.col_widths:
            return self.col_widths
        
        widths = [len(h) for h in self.headers]
        for row in self.rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))
        
        return widths
    
    def render(self) -> str:
        """渲染表格为字符串"""
        widths = self._calculate_widths()
        lines = []
        sep = " " * self.padding
        
        # 表头
        header_line = sep.join(
            h.ljust(w) for h, w in zip(self.headers, widths)
        )
        lines.append(f"  {header_line}")
        
        # 分隔线
        total_width = sum(widths) + self.padding * (len(widths) - 1)
        lines.append("  " + "─" * total_width)
        
        # 数据行
        for i, row in enumerate(self.rows):
            cells = []
            for j, (cell, width) in enumerate(zip(row, widths)):
                if j < len(row):
                    cells.append(cell.ljust(width))
            
            line = sep.join(cells)
            if i == self.highlight_row:
                lines.append(f"→ {line}")
            else:
                lines.append(f"  {line}")
        
        return "\n".join(lines)
    
    def print(self):
        """打印表格"""
        print(self.render())


# ============================================================================
# Panel 组件
# ============================================================================

class Panel:
    """面板/卡片组件"""
    
    def __init__(self, title: str = "", width: int = 40):
        """
        初始化面板
        
        Args:
            title: 面板标题
            width: 面板宽度
        """
        self.title = title
        self.width = width
        self.lines: List[str] = []
    
    def add_line(self, key: str, value: str):
        """添加键值对行"""
        self.lines.append((key, value))
    
    def add_separator(self):
        """添加分隔线"""
        self.lines.append(None)
    
    def render(self) -> str:
        """渲染面板为字符串"""
        inner_width = self.width - 4  # 边框占用
        result = []
        
        # 顶部边框
        result.append("  ┌" + "─" * (self.width - 2) + "┐")
        
        # 标题（如果有）
        if self.title:
            title_line = f" {self.title} ".center(self.width - 2)
            result.append("  │" + title_line + "│")
            result.append("  ├" + "─" * (self.width - 2) + "┤")
        
        # 内容行
        for line in self.lines:
            if line is None:
                # 分隔线
                result.append("  ├" + "─" * (self.width - 2) + "┤")
            else:
                key, value = line
                # 计算可用空间
                content = f" {key:<12} {value}"
                if len(content) > inner_width:
                    content = content[:inner_width]
                content = content.ljust(inner_width)
                result.append("  │ " + content + " │")
        
        # 底部边框
        result.append("  └" + "─" * (self.width - 2) + "┘")
        
        return "\n".join(result)
    
    def print(self):
        """打印面板"""
        print(self.render())


# ============================================================================
# ConfigCard 组件
# ============================================================================

class ConfigCard:
    """配置摘要卡片"""
    
    def __init__(self, config_dict: Dict[str, str], title: str = "当前配置"):
        """
        初始化配置卡片
        
        Args:
            config_dict: 配置字典 {显示名: 值}
            title: 卡片标题
        """
        self.config = config_dict
        self.title = title
    
    def render(self) -> str:
        """渲染配置卡片"""
        lines = []
        width = 42
        
        lines.append("")
        lines.append(f"  {self.title}:")
        lines.append("")
        
        # 找出最长的键名
        max_key_len = max(len(k) for k in self.config.keys())
        
        for key, value in self.config.items():
            lines.append(f"    {key:<{max_key_len}}  {value}")
        
        lines.append("")
        return "\n".join(lines)
    
    def print(self):
        """打印配置卡片"""
        print(self.render())


# ============================================================================
# Spinner 组件
# ============================================================================

class Spinner:
    """加载旋转器"""
    
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FRAMES_ASCII = ["|", "/", "-", "\\"]
    
    def __init__(self, message: str = "加载中", use_unicode: bool = None):
        """
        初始化旋转器
        
        Args:
            message: 显示消息
            use_unicode: 是否使用 Unicode 字符，None 表示自动检测
        """
        self.message = message
        self._running = False
        self._thread = None
        
        # 自动检测是否支持 Unicode
        if use_unicode is None:
            use_unicode = self._supports_unicode()
        
        self.frames = self.FRAMES if use_unicode else self.FRAMES_ASCII
    
    @staticmethod
    def _supports_unicode() -> bool:
        """检测终端是否支持 Unicode"""
        if platform.system() == 'Windows':
            # Windows 控制台可能不支持 Unicode
            try:
                sys.stdout.write("▪")
                sys.stdout.write("\b \b")
                return True
            except (UnicodeEncodeError, Exception):
                return False
        return True
    
    def _spin(self):
        """旋转动画线程"""
        i = 0
        while self._running:
            frame = self.frames[i % len(self.frames)]
            sys.stdout.write(f"\r  {frame} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        # 清除行
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()
    
    def start(self):
        """开始旋转"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    
    def stop(self, final_message: str = None):
        """停止旋转"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if final_message:
            print(f"  {final_message}")


# ============================================================================
# ProgressBar 组件
# ============================================================================

class ProgressBar:
    """进度条"""
    
    def __init__(self, total: int, width: int = 30, 
                 fill_char: str = "█", empty_char: str = "░"):
        """
        初始化进度条
        
        Args:
            total: 总步数
            width: 进度条宽度
            fill_char: 填充字符
            empty_char: 空白字符
        """
        self.total = total
        self.current = 0
        self.width = width
        self.fill_char = fill_char
        self.empty_char = empty_char
    
    def update(self, current: int, description: str = ""):
        """更新进度"""
        self.current = current
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * percent)
        
        bar = self.fill_char * filled + self.empty_char * (self.width - filled)
        line = f"\r  [{bar}] {self.current}/{self.total}"
        if description:
            line += f" {description}"
        
        sys.stdout.write(line)
        sys.stdout.flush()
        
        if self.current >= self.total:
            print()  # 完成时换行
    
    def increment(self, description: str = ""):
        """增加进度"""
        self.update(self.current + 1, description)


# ============================================================================
# Menu 组件
# ============================================================================

class MenuItem:
    """菜单项"""
    
    def __init__(self, key: str, label: str, 
                 description: str = "", 
                 handler: Callable = None,
                 icon: str = ""):
        self.key = key
        self.label = label
        self.description = description
        self.handler = handler
        self.icon = icon


class Menu:
    """交互式菜单"""
    
    def __init__(self, title: str, icon: str = ""):
        """
        初始化菜单
        
        Args:
            title: 菜单标题
            icon: 菜单图标
        """
        self.title = title
        self.icon = icon
        self.items: List[MenuItem] = []
        self.groups: Dict[str, List[MenuItem]] = {}
        self.current_group: str = ""
    
    def add_group(self, name: str):
        """添加分组"""
        self.current_group = name
        self.groups[name] = []
    
    def add_item(self, key: str, label: str, 
                 handler: Callable = None,
                 description: str = "",
                 icon: str = ""):
        """添加菜单项"""
        item = MenuItem(key, label, description, handler, icon)
        self.items.append(item)
        
        if self.current_group:
            self.groups[self.current_group].append(item)
    
    def render(self) -> str:
        """渲染菜单"""
        lines = []
        
        # 按分组渲染
        if self.groups:
            for group_name, group_items in self.groups.items():
                lines.append(f"  ─── {group_name} ───")
                for item in group_items:
                    icon = f"{item.icon} " if item.icon else ""
                    lines.append(f"  [{item.key}] {icon}{item.label}")
                    if item.description:
                        lines.append(f"      {item.description}")
                lines.append("")
        else:
            # 无分组
            for item in self.items:
                icon = f"{item.icon} " if item.icon else ""
                lines.append(f"  [{item.key}] {icon}{item.label}")
                if item.description:
                    lines.append(f"      {item.description}")
        
        return "\n".join(lines)
    
    def print(self):
        """打印菜单"""
        print_menu_header(self.title, self.icon)
        print(self.render())
    
    def get_valid_keys(self) -> List[str]:
        """获取所有有效的选择键"""
        return [item.key for item in self.items]
    
    def handle_choice(self, choice: str) -> bool:
        """
        处理用户选择
        
        Args:
            choice: 用户选择的键
            
        Returns:
            是否找到并执行了处理函数
        """
        for item in self.items:
            if item.key.lower() == choice.lower():
                if item.handler:
                    item.handler()
                    return True
                return False
        return False


# ============================================================================
# SelectList 组件
# ============================================================================

class SelectList:
    """选择列表"""
    
    def __init__(self, title: str, items: List[tuple], 
                 allow_multi: bool = False):
        """
        初始化选择列表
        
        Args:
            title: 列表标题
            items: 选项列表 [(value, display_name), ...]
            allow_multi: 是否允许多选
        """
        self.title = title
        self.items = items
        self.allow_multi = allow_multi
    
    def render(self) -> str:
        """渲染选择列表"""
        lines = [f"\n{self.title}:"]
        
        for i, (value, display) in enumerate(self.items, 1):
            lines.append(f"  [{i}] {display}")
        
        return "\n".join(lines)
    
    def print(self):
        """打印选择列表"""
        print(self.render())
    
    def get_selection(self, prompt: str = "请选择: ") -> Any:
        """
        获取用户选择
        
        Returns:
            单选返回值，多选返回值列表
        """
        self.print()
        
        if self.allow_multi:
            raw = input(f"{prompt}(用空格分隔多个选项): ").strip()
            indices = self._parse_multi_choice(raw)
            return [self.items[i-1][0] for i in indices if 1 <= i <= len(self.items)]
        else:
            valid = [str(i) for i in range(1, len(self.items) + 1)]
            choice = get_choice(prompt, valid)
            idx = int(choice) - 1
            return self.items[idx][0]
    
    def _parse_multi_choice(self, input_str: str) -> List[int]:
        """解析多选输入"""
        choices = []
        for part in input_str.replace(",", " ").split():
            try:
                val = int(part)
                if val not in choices:
                    choices.append(val)
            except ValueError:
                pass
        return sorted(choices) if choices else [1]