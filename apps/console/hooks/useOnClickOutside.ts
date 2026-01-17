import { RefObject, useEffect } from 'react';

type Handler = (event: MouseEvent | TouchEvent) => void;

/**
 * Custom hook to detect clicks outside of a specified element.
 * @param ref The ref of the element to detect clicks outside of.
 * @param handler The function to call when a click outside is detected.
 */
export function useOnClickOutside<T extends HTMLElement = HTMLElement>(
  ref: RefObject<T>,
  handler: Handler,
): void {
  useEffect(
    () => {
      const listener = (event: MouseEvent | TouchEvent) => {
        const el = ref?.current;

        // Do nothing if clicking ref's element or descendent elements
        if (!el || el.contains(event.target as Node)) {
          return;
        }

        handler(event);
      };

      document.addEventListener('mousedown', listener);
      document.addEventListener('touchstart', listener);

      return () => {
        document.removeEventListener('mousedown', listener);
        document.removeEventListener('touchstart', listener);
      };
    },
    // Add ref and handler to effect dependencies
    // It's worth noting that because passed in handler is a new function on every render that will cause this effect callback/cleanup to run every render.
    // It's not a big deal but to optimize you can wrap handler in useCallback before passing it into this hook.
    [ref, handler],
  );
}
