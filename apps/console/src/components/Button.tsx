import React from 'react';
import { twMerge } from 'tailwind-merge';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
}

export const Button: React.FC<ButtonProps> = ({
  className,
  variant = 'primary',
  size = 'md',
  children,
  ...props
}) => {
  const baseStyles = "inline-flex items-center justify-center rounded transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gothic-900 font-medium";

  const variants = {
    primary: "bg-gothic-gold text-gothic-900 hover:bg-yellow-600 focus:ring-gothic-gold",
    secondary: "bg-gothic-700 text-gothic-text hover:bg-gothic-600 focus:ring-gothic-700 border border-gothic-600",
    ghost: "bg-transparent hover:bg-gothic-800 text-gothic-muted hover:text-gothic-text",
    danger: "bg-red-900/50 text-red-200 border border-red-900 hover:bg-red-900 focus:ring-red-500",
  };

  const sizes = {
    sm: "px-3 py-1.5 text-xs",
    md: "px-4 py-2 text-sm",
    lg: "px-6 py-3 text-base",
  };

  return (
    <button
      className={twMerge(baseStyles, variants[variant], sizes[size], className)}
      {...props}
    >
      {children}
    </button>
  );
};
