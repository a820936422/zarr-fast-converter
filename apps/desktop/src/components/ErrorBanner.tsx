import { Icon } from "./Icon";

export function ErrorBanner({ message, onClose }: { message: string | null; onClose: () => void }) {
  return (
    <div className="error-toast" role="alert">
      <Icon name="terminal" size={17} />
      <span>{message}</span>
      <button type="button" onClick={onClose}>×</button>
    </div>
  );
}