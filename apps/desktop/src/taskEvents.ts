import { useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { cancelTask, getTask, listTasks, type TaskEvent, type TaskSummary } from "./api";

export function useTaskEvents() {
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [tasks, setTasks] = useState<TaskSummary[]>([]);

  useEffect(() => {
    let active = true;
    const refreshTasks = () => {
      void listTasks().then((result) => {
        if (active) setTasks(result);
      }).catch(() => undefined);
    };
    refreshTasks();
    window.addEventListener("focus", refreshTasks);
    let unlisten: (() => void) | undefined;
    void listen<TaskEvent>("task-event", (event) => {
      if (!active) return;
      const payload = event.payload;
      setEvents((current) => [...current.slice(-199), payload]);
      if (payload.task_id) {
        void getTask(payload.task_id).then((task) => {
          if (task && active) setTasks((current) => [...current.filter((item) => item.taskId !== task.taskId), task]);
        }).catch(() => undefined);
      }
    }).then((dispose) => { unlisten = dispose; }).catch(() => undefined);
    return () => {
      active = false;
      window.removeEventListener("focus", refreshTasks);
      unlisten?.();
    };
  }, []);

  return { events, tasks, cancel: cancelTask };
}
