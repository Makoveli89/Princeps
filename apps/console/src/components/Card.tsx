import React from 'react';
import { twMerge } from 'tailwind-merge';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glow?: boolean;
}

export const Card: React.FC<CardProps> = ({ className, glow = false, children, ...props }) => {
  return (
    <div
      className={twMerge(
        "bg-gothic-800 border border-gothic-700 rounded-lg p-6",
        glow && "shadow-[0_0_15px_rgba(109,40,217,0.15)] border-gothic-purple/30",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
