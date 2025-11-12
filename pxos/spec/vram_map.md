# VRAM Region Map for pxOS

This document specifies the spatial layout of memory regions within a standard 1024x1024 pxOS VRAM texture.

## Memory Layout

The VRAM is divided into four primary regions, each with a distinct purpose and coordinate range.

```
(0,0) ┌────────────────────────────────────────────────────────┐ (1024,0)
      │                                                        │
      │                  META REGION                           │
      │          (Process Tables, Syscall Vector)              │
      │                  [0-256, 0-256]                        │
      ├────────────────────────────────────────────────────────┤
      │                                                        │
      │                  PXI REGION                            │
      │           (Executable Program Image)                   │
      │                 [0-768, 256-768]                       │
      │                                                        │
      ├────────────────────────────────────────────────────────┤
      │            DATA REGION              │   FRAME REGION   │
      │         (Working Memory)            │  (Display Output)│
      │          [0-512, 768-1024]          │ [512-1024,768-1024]│
      └────────────────────────────────────────────────────────┘
   (0,1024)                                                 (1024,1024)
```

## Region Details

| Region Name | Coordinate Range | Color Code | Purpose |
| :--- | :--- | :--- | :--- |
| **META** | `[0-256, 0-256]` | Light Blue | Process Tables, Syscall Vectors, Permissions |
| **PXI** | `[0-768, 256-768]` | Multi-colored | Executable Program Image (Instructions) |
| **DATA** | `[0-512, 768-1024]` | Green Gradient | Working Memory, Registers, Buffers |
| **FRAME** | `[512-1024, 768-1024]` | Black background | Final Visual Output for Display |

## Memory Hierarchy Flow

The typical data flow between regions is as follows:

1. The **Kernel Cursor** reads instructions from the **PXI REGION**.
2. Instructions operate on data within the **DATA REGION**.
3. System-level information is accessed from the **META REGION**.
4. The `DRAW` instruction copies the final output from the **DATA REGION** to the **FRAME REGION** for display.
