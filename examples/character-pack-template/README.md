# Character pack template

This folder is directly importable through **Dashboard → Avatar settings →
Import character ZIP**. Before distributing it:

1. Change `id`, `name`, `author`, and `license` in `pack.yaml`.
2. Replace the sample transparent PNG in every state folder.
3. Add numbered frames (`000.png`, `001.png`, …) for animation.
4. Add optional `<state>_speaking` folders and point `speaking_frames` to them.
5. ZIP this folder so the archive contains exactly one `pack.yaml`.

Pack IDs use lowercase letters, numbers, hyphens, and underscores. Import does
not overwrite an existing ID.
